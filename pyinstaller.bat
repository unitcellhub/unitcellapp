@ECHO OFF
:: Build pyinstaller executable

uv sync --frozen
uv run pyinstaller --noconfirm pyinstaller.spec
