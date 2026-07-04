from cyclopts import App
from utils4plans.logconfig import logset

from nvml.cli.smk.qdim_data import qdim

app = App()
app.command(qdim)


def main():
    logset(to_stderr=True)
    app()


if __name__ == "__main__":
    main()
