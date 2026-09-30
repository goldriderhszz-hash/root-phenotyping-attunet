"""Thread-local presentation language; scientific field names are never translated."""
from contextvars import ContextVar
import json
import os
from pathlib import Path

LANGUAGES = ('en', 'zh')
_language = ContextVar('rootscope_language', default='en')
MESSAGES = json.loads(Path(__file__).with_name('translations.json').read_text(encoding='utf-8'))

def set_language(language):
    if language not in LANGUAGES:
        raise ValueError(f'Unsupported language: {language}')
    _language.set(language)

def get_language():
    return _language.get()

def tr(key, **values):
    return MESSAGES[key][get_language()].format(**values)

def preferences_path():
    root = Path(os.environ.get('LOCALAPPDATA', Path.home() / '.config'))
    return root / 'RootScope' / 'preferences.json'

def load_language():
    try:
        language = json.loads(preferences_path().read_text(encoding='utf-8'))['language']
        return language if language in LANGUAGES else 'en'
    except (OSError, ValueError, KeyError, TypeError):
        return 'en'

def save_language(language):
    if language not in LANGUAGES:
        raise ValueError(language)
    try:
        path = preferences_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'language': language}), encoding='utf-8')
    except OSError:
        # Read-only profiles still permit an in-session language change.
        pass
