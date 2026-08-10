# class AlgoFactory:
#     def __init__(self):
#         self._builders = {}
#         self._configs = {}
#
#     def register_builder(self, key, builder, config):
#         self._builders[key] = builder
#         self._configs[key] = config
#
#     def create(self, key, **kwargs):
#         builder = self._builders.get(key)
#         if not builder:
#             raise ValueError(key)
#         return builder(**kwargs)
#
#     def initialize(self, item):
#         print("Adding algorithms...")
#         for key in self._builders:
#             item.addItem(key)
#             print(key)
