# HackThe6ix Voice Bot

## Getting Started

To run `voice_bot.py`, follow these steps in your terminal:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

This will set up a virtual environment and install all required dependencies.

## Notes

- Make sure you are in the project directory before running these commands.
- If you are on Windows, use `venv\Scripts\activate` instead of `source venv/bin/activate`.


## Changes to README.md
- MacOS doesn't require pypiwin32 or pywin32
- If in Mac and getting Failed to build PyAudio - try running the command below
```bash
brew install portaudio
```