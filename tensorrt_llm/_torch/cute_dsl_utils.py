import platform

from ..logger import logger

IS_CUTLASS_DSL_AVAILABLE = False
# TODO: Remove IS_CUTLASS_DSL_INTERNAL_AVAILABLE once rubin_helpers is available
# in the base nvidia-cutlass-dsl package (currently only in nvidia-cutlass-dsl-internal)
IS_CUTLASS_DSL_INTERNAL_AVAILABLE = False

if platform.system() != "Windows":
    try:
        import cutlass  # noqa
        import cutlass.cute as cute  # noqa
        logger.info(f"cutlass dsl is available")
        IS_CUTLASS_DSL_AVAILABLE = True

        # Check for internal cutlass DSL package (has rubin_helpers for SM107)
        try:
            import cutlass.utils.rubin_helpers  # noqa
            logger.info(f"cutlass dsl internal (rubin) is available")
            IS_CUTLASS_DSL_INTERNAL_AVAILABLE = True
        except ImportError:
            pass
    except ImportError:
        pass
