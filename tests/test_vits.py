import unittest

from multivit import config, modules


class TestMultiVitComponent(unittest.TestCase):

    def test_multivit_component_init(self):

        m = modules.MultiViTComponent(config=config.MultiViTComponentConfig())
