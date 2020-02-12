import platform
import sys
import os
import re
import warnings
import multiprocessing

# YAML needed to use file based Numba config
try:
    import yaml
    _HAVE_YAML = True
except ImportError:
    _HAVE_YAML = False


import llvmlite.binding as ll

IS_WIN32 = sys.platform.startswith('win32')
IS_OSX = sys.platform.startswith('darwin')
MACHINE_BITS = tuple.__itemsize__ * 8
IS_32BITS = MACHINE_BITS == 32
# Python version in (major, minor) tuple
PYVERSION = sys.version_info[:2]

# this is the name of the user supplied configuration file
_config_fname = '.numba_config.yaml'


def _parse_cc(text):
    """
    Parse CUDA compute capability version string.
    """
    if not text:
        return None
    else:
        m = re.match(r'(\d+)\.(\d+)', text)
        if not m:
            raise ValueError("NUMBA_FORCE_CUDA_CC must be specified as a "
                             "string of \"major.minor\" where major "
                             "and minor are decimals")
        grp = m.groups()
        return int(grp[0]), int(grp[1])


def _os_supports_avx():
    """
    Whether the current OS supports AVX, regardless of the CPU.

    This is necessary because the user may be running a very old Linux
    kernel (e.g. CentOS 5) on a recent CPU.
    """
    if (not sys.platform.startswith('linux')
            or platform.machine() not in ('i386', 'i586', 'i686', 'x86_64')):
        return True
    # Executing the CPUID instruction may report AVX available even though
    # the kernel doesn't support it, so parse /proc/cpuinfo instead.
    try:
        f = open('/proc/cpuinfo', 'r')
    except OSError:
        # If /proc isn't available, assume yes
        return True
    with f:
        for line in f:
            head, _, body = line.partition(':')
            if head.strip() == 'flags' and 'avx' in body.split():
                return True
        else:
            return False


class _EnvReloader(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.old_environ = {}
        self.update(force=True)

    def update(self, force=False):
        new_environ = {}

        # first check if there's a .numba_config.yaml and use values from that
        if os.path.exists(_config_fname) and os.path.isfile(_config_fname):
            if not _HAVE_YAML:
                msg = ("A Numba config file is found but YAML parsing "
                       "capabilities appear to be missing. "
                       "To use this feature please install `pyyaml`. e.g. "
                       "`conda install pyyaml`.")
                warnings.warn(msg)
            else:
                with open(_config_fname, 'rt') as f:
                    y_conf = yaml.safe_load(f)
                if y_conf is not None:
                    for k, v in y_conf.items():
                        new_environ['NUMBA_' + k.upper()] = v

        # clobber file based config with any locally defined env vars
        for name, value in os.environ.items():
            if name.startswith('NUMBA_'):
                new_environ[name] = value
        # We update the config variables if at least one NUMBA environment
        # variable was modified.  This lets the user modify values
        # directly in the config module without having them when
        # reload_config() is called by the compiler.
        if force or self.old_environ != new_environ:
            self.process_environ(new_environ)
            # Store a copy
            self.old_environ = dict(new_environ)

    def process_environ(self, environ):
        def _readenv(name, ctor, default, allow_reload=True,
                     reload_error_message=None):
            """
            Read name from the environment as NUMBA_name with the type ctor and
            value default.

            If allow_reload is False and the variable has already been loaded
            with a different value, RuntimeError with reload_error_message is
            raised. allow_reload can also be a callable that returns a boolean
            to allow delayed evaluation.

            reload_error_message can have the format variables {name} for the
            config name (without "NUMBA_"), {current} to for the already
            loaded value, and {new} to represent the newly loaded value.

            """
            value = environ.get("NUMBA_" + name)
            default_value = default() if callable(default) else default
            if not callable(allow_reload):
                _allow_reload = allow_reload
                allow_reload = lambda: _allow_reload

            if (name in globals()
                and globals()[name] != default_value
                and allow_reload()):

                if not reload_error_message:
                    reload_error_message = (
                        "Cannot set NUMBA_{name} to a different value once it "
                        "has already been loaded (currently set to {name}, "
                        "trying to set it to {new})")

                raise RuntimeError(reload_error_message.format(
                    name=name, current=globals()[name], new=default))

            if value is None:
                return default_value

            try:
                return ctor(value)
            except Exception:
                warnings.warn("environ %s defined but failed to parse '%s'" %
                              (name, value), RuntimeWarning)
                return default_value

        def optional_str(x):
            return str(x) if x is not None else None

        # developer mode produces full tracebacks, disables help instructions
        DEVELOPER_MODE = _readenv("DEVELOPER_MODE", int, 0)

        # disable performance warnings, will switch of the generation of
        # warnings of the class NumbaPerformanceWarning
        DISABLE_PERFORMANCE_WARNINGS = _readenv(
            "DISABLE_PERFORMANCE_WARNINGS", int, 0)

        # Flag to enable full exception reporting
        FULL_TRACEBACKS = _readenv(
            "FULL_TRACEBACKS", int, DEVELOPER_MODE)

        # Show help text when an error occurs
        SHOW_HELP = _readenv("SHOW_HELP", int, 0)

        # The color scheme to use for error messages, default is no color
        # just bold fonts in use.
        COLOR_SCHEME = _readenv("COLOR_SCHEME", str, "no_color")

        # Whether to globally enable bounds checking. The default None means
        # to use the value of the flag to @njit. 0 or 1 overrides the flag
        # globally.
        BOUNDSCHECK = _readenv("BOUNDSCHECK", int, None)

        # Debug flag to control compiler debug print
        DEBUG = _readenv("DEBUG", int, 0)

        # DEBUG print IR after pass names
        DEBUG_PRINT_AFTER = _readenv("DEBUG_PRINT_AFTER", str, "none")

        # DEBUG print IR before pass names
        DEBUG_PRINT_BEFORE = _readenv("DEBUG_PRINT_BEFORE", str, "none")

        # DEBUG print IR before and after pass names
        DEBUG_PRINT_WRAP = _readenv("DEBUG_PRINT_WRAP", str, "none")

        # Highlighting in intermediate dumps
        HIGHLIGHT_DUMPS = _readenv("HIGHLIGHT_DUMPS", int, 0)

        # JIT Debug flag to trigger IR instruction print
        DEBUG_JIT = _readenv("DEBUG_JIT", int, 0)

        # Enable debugging of front-end operation
        # (up to and including IR generation)
        DEBUG_FRONTEND = _readenv("DEBUG_FRONTEND", int, 0)

        # How many recently deserialized functions to retain regardless
        # of external references
        FUNCTION_CACHE_SIZE = _readenv("FUNCTION_CACHE_SIZE", int, 128)

        # Enable logging of cache operation
        DEBUG_CACHE = _readenv("DEBUG_CACHE", int, DEBUG)

        # Redirect cache directory
        # Contains path to the directory
        CACHE_DIR = _readenv("CACHE_DIR", str, "")

        # Enable tracing support
        TRACE = _readenv("TRACE", int, 0)

        # Enable debugging of type inference
        DEBUG_TYPEINFER = _readenv("DEBUG_TYPEINFER", int, 0)

        # Configure compilation target to use the specified CPU name
        # and CPU feature as the host information.
        # Note: this overrides "host" option for AOT compilation.
        CPU_NAME = _readenv("CPU_NAME", optional_str, None)
        CPU_FEATURES = _readenv("CPU_FEATURES", optional_str,
                                ("" if str(CPU_NAME).lower() == 'generic'
                                 else None))
        # Optimization level
        OPT = _readenv("OPT", int, 3)

        # Force dump of Python bytecode
        DUMP_BYTECODE = _readenv("DUMP_BYTECODE", int, DEBUG_FRONTEND)

        # Force dump of control flow graph
        DUMP_CFG = _readenv("DUMP_CFG", int, DEBUG_FRONTEND)

        # Force dump of Numba IR
        DUMP_IR = _readenv("DUMP_IR", int,
                           DEBUG_FRONTEND or DEBUG_TYPEINFER)

        # print debug info of analysis and optimization on array operations
        DEBUG_ARRAY_OPT = _readenv("DEBUG_ARRAY_OPT", int, 0)

        # insert debug stmts to print information at runtime
        DEBUG_ARRAY_OPT_RUNTIME = _readenv(
            "DEBUG_ARRAY_OPT_RUNTIME", int, 0)

        # print stats about parallel for-loops
        DEBUG_ARRAY_OPT_STATS = _readenv("DEBUG_ARRAY_OPT_STATS", int, 0)

        # prints user friendly information about parallel
        PARALLEL_DIAGNOSTICS = _readenv("PARALLEL_DIAGNOSTICS", int, 0)

        # print debug info of inline closure pass
        DEBUG_INLINE_CLOSURE = _readenv("DEBUG_INLINE_CLOSURE", int, 0)

        # Force dump of LLVM IR
        DUMP_LLVM = _readenv("DUMP_LLVM", int, DEBUG)

        # Force dump of Function optimized LLVM IR
        DUMP_FUNC_OPT = _readenv("DUMP_FUNC_OPT", int, DEBUG)

        # Force dump of Optimized LLVM IR
        DUMP_OPTIMIZED = _readenv("DUMP_OPTIMIZED", int, DEBUG)

        # Force disable loop vectorize
        # Loop vectorizer is disabled on 32-bit win32 due to a bug (#649)
        LOOP_VECTORIZE = _readenv("LOOP_VECTORIZE", int,
                                  not (IS_WIN32 and IS_32BITS))

        # Force dump of generated assembly
        DUMP_ASSEMBLY = _readenv("DUMP_ASSEMBLY", int, DEBUG)

        # Force dump of type annotation
        ANNOTATE = _readenv("DUMP_ANNOTATION", int, 0)

        # Dump IR in such as way as to aid in "diff"ing.
        DIFF_IR = _readenv("DIFF_IR", int, 0)

        # Dump type annotation in html format
        def fmt_html_path(path):
            if path is None:
                return path
            else:
                return os.path.abspath(path)

        HTML = _readenv("DUMP_HTML", fmt_html_path, None)

        # Allow interpreter fallback so that Numba @jit decorator will never
        # fail. Use for migrating from old numba (<0.12) which supported
        # closure, and other yet-to-be-supported features.
        COMPATIBILITY_MODE = _readenv("COMPATIBILITY_MODE", int, 0)

        # x86-64 specific
        # Enable AVX on supported platforms where it won't degrade performance.
        def avx_default():
            if not _os_supports_avx():
                return False
            else:
                # There are various performance issues with AVX and LLVM
                # on some CPUs (list at
                # http://llvm.org/bugs/buglist.cgi?quicksearch=avx).
                # For now we'd rather disable it, since it can pessimize code
                cpu_name = ll.get_host_cpu_name()
                return cpu_name not in ('corei7-avx', 'core-avx-i',
                                        'sandybridge', 'ivybridge')

        ENABLE_AVX = _readenv("ENABLE_AVX", int, avx_default)

        # if set and SVML is available, it will be disabled
        # By default, it's disabled on 32-bit platforms.
        DISABLE_INTEL_SVML = _readenv(
            "DISABLE_INTEL_SVML", int, IS_32BITS)

        # Disable jit for debugging
        DISABLE_JIT = _readenv("DISABLE_JIT", int, 0)

        # choose parallel backend to use
        THREADING_LAYER = _readenv("THREADING_LAYER", str, 'default')

        # CUDA Configs

        # Force CUDA compute capability to a specific version
        FORCE_CUDA_CC = _readenv("FORCE_CUDA_CC", _parse_cc, None)

        # Disable CUDA support
        DISABLE_CUDA = _readenv("DISABLE_CUDA",
                                int, int(MACHINE_BITS == 32))

        # Enable CUDA simulator
        ENABLE_CUDASIM = _readenv("ENABLE_CUDASIM", int, 0)

        # CUDA logging level
        # Any level name from the *logging* module.  Case insensitive.
        # Defaults to CRITICAL if not set or invalid.
        # Note: This setting only applies when logging is not configured.
        #       Any existing logging configuration is preserved.
        CUDA_LOG_LEVEL = _readenv("CUDA_LOG_LEVEL", str, '')

        # Maximum number of pending CUDA deallocations (default: 10)
        CUDA_DEALLOCS_COUNT = _readenv("CUDA_MAX_PENDING_DEALLOCS_COUNT",
                                       int, 10)

        # Maximum ratio of pending CUDA deallocations to capacity (default: 0.2)
        CUDA_DEALLOCS_RATIO = _readenv("CUDA_MAX_PENDING_DEALLOCS_RATIO",
                                       float, 0.2)

        # HSA Configs

        # Disable HSA support
        DISABLE_HSA = _readenv("DISABLE_HSA", int, 0)

        # The default number of threads to use.
        NUMBA_DEFAULT_NUM_THREADS = max(1, multiprocessing.cpu_count())

        # Numba thread pool size (defaults to number of CPUs on the system).
        def _parallel_not_initialized():
            from numba.np.ufunc import parallel
            return not parallel._is_initialized

        NUMBA_NUM_THREADS = _readenv("NUM_THREADS", int,
                                     NUMBA_DEFAULT_NUM_THREADS,
                                     allow_reload=_parallel_not_initialized,
                                     reload_error_message=("""\
Cannot set NUMBA_{name} to a different value once threads have been launched \
(currently set to {name}, trying to set it to {new})")"""))

        # Profiling support

        # Indicates if a profiler detected. Only VTune can be detected for now
        RUNNING_UNDER_PROFILER = 'VS_PROFILER' in os.environ

        # Enables jit events in LLVM to support profiling of dynamic code
        ENABLE_PROFILING = _readenv(
            "ENABLE_PROFILING", int, int(RUNNING_UNDER_PROFILER))

        # Debug Info

        # The default value for the `debug` flag
        DEBUGINFO_DEFAULT = _readenv("DEBUGINFO", int, ENABLE_PROFILING)
        CUDA_DEBUGINFO_DEFAULT = _readenv("CUDA_DEBUGINFO", int, 0)

        # gdb binary location
        GDB_BINARY = _readenv("GDB_BINARY", str, '/usr/bin/gdb')

        # Inject the configuration values into the module globals
        for name, value in locals().copy().items():
            if name.isupper():
                globals()[name] = value


_env_reloader = _EnvReloader()


def reload_config():
    """
    Reload the configuration from environment variables, if necessary.
    """
    _env_reloader.update()
