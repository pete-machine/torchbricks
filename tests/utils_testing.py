
import functools
import inspect
from pathlib import Path
import re


def path_repo_root():
    return Path(__file__).parents[1]


def copy_codesnippet_to_readme(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        value = func(*args, **kwargs)
        path_repo = path_repo_root()
        path_readme = path_repo / 'README.md'
        with open(path_readme) as file:
            readme_content = file.read()

        # Looks for in README.md "<!--- some_random_chars @get_source_code([SOME_NAME]) some_random_chars --->"
        readme_cmd = f'<!--- @get_source_code({func.__name__}) - below code snippet is automatically taken from the specified test --->'
        pattern = r'<!---.*?@get_source_code\(' + func.__name__ + r'\).*?--->'
        match = re.search(pattern, readme_content)
        print(match)
        if match:
            code_snippet = inspect.getsource(func)
            # Split lines and drop the first two
            lines = code_snippet.split('\n')
            lines = lines[2:]  # Drop first two lines
            lines = [line[4:] for line in lines]  # Remove indent
            lines.insert(0, '```py')
            lines.insert(0, readme_cmd)
            lines.append('```')
            readme_replace = '\n'.join(lines)

        else:
            raise ValueError(f"Failed to copy code snippet of '{func.__name__}' into '{path_readme}'. "
                             f"Could not to find '<!--- get_source_code({func.__name__}) -->' in README.md. "
                             f"If the code snippet in '{func.__name__}' is no longer used in the README.md - you can delete the test!")

        pattern_readme_already_has_snippet = pattern + r'.*?\n```py'
        has_code_snippet = re.search(pattern_readme_already_has_snippet, readme_content)
        if has_code_snippet:
            # Replace also code snippet in README.md
            pattern_with_script = pattern + r'\n```py.*?```'
            readme_content = re.sub(pattern_with_script, readme_replace, readme_content, flags=re.DOTALL)
        else:
            # Replace only comment in README.md
            readme_content = re.sub(pattern, readme_replace, readme_content, flags=re.DOTALL)

        with open(path_readme, mode='w') as file:
            file.write(readme_content)

        return value
    return wrapper_decorator
