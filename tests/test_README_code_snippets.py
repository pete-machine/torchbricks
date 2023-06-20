
## IF ANY TESTS FAILS - YOU SHOULD UPDATE THE README
import re
import torch
from torch import nn
import inspect
import functools

import utils_testing
def copy_codesnippet_to_readme(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        value = func(*args, **kwargs)
        path_repo = utils_testing.path_repo_root()
        path_readme = path_repo / 'README.md'
        with open(path_readme) as file:
            readme_content = file.read()

        # Looks for in README.md "<!--- some_random_chars @get_source_code([SOME_NAME]) some_random_chars --->"
        readme_cmd = f'<!--- @get_source_code({func.__name__}) --->'
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

        with open(path_readme, 'w') as file:
            file.write(readme_content)

        return value
    return wrapper_decorator

class Preprocessor(nn.Module):
    def forward(self, raw_input: torch.Tensor) -> torch.Tensor:
        return raw_input/2

class TinyModel(nn.Module):
    def __init__(self, n_channels: int, n_features: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_features, 1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.conv(tensor)

class Classifier(nn.Module):
    def __init__(self, num_classes: int, in_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.flatten(self.avgpool(tensor)))


@copy_codesnippet_to_readme
def test_basic_use_case_image_classification():
    from torchbricks.bricks import BrickCollection, BrickNotTrainable, BrickTrainable, Phase

    # Defining model from bricks
    bricks = {
        'preprocessor': BrickNotTrainable(Preprocessor(), input_names=['raw'], output_names=['processed']),
        'backbone': BrickTrainable(TinyModel(n_channels=3, n_features=10), input_names=['processed'], output_names=['embedding']),
        'classifier': BrickTrainable(Classifier(num_classes=3, in_features=10), input_names=['embedding'], output_names=['logits'])
    }

    # Executing model
    model = BrickCollection(bricks)
    batch_image_example = torch.rand((1, 3, 100, 200))
    outputs = model(named_inputs={'raw': batch_image_example}, phase=Phase.TRAIN)
    print(outputs.keys())
