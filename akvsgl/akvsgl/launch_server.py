"""Launch AKV through SGLang's native server entry point."""

import runpy


if __name__ == "__main__":
    runpy.run_module("sglang.launch_server", run_name="__main__", alter_sys=True)
