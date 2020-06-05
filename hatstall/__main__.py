import argparse
import importlib


parser = argparse.ArgumentParser(description='Personality classifier')
parser.add_argument(
    '--mode', default='train_test',
    help=(
        'Enter the mode in which the personality classifier should be run. '
        'Select from: [train_test, predict]'))


class AttributeDict(dict):
    """A dictionary that makes its keys available as attributes"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def copy(self):
        return AttributeDict(super().copy())


def load_settings():
    settings = importlib.import_module('hatstall.settings')
    return AttributeDict(
       {k: v for k, v in vars(settings).items() if k.isupper()})


def parse_args(settings, args):
    settings['MODE'] = args.mode


def main(settings):
    pl_system = settings.PIPELINE_SYSTEM
    pl_system.run()


if __name__ == '__main__':
    args = parser.parse_args()
    settings = load_settings()
    parse_args(settings, args)
    main(settings)
