import json

class Param():
	def __init__(self, json_path):
		self.update(json_path)

	def save(self, json_path):
		with open(json_path, 'w') as f:
			f.dump(self.__dict__, f, indent=4)

	def update(self, json_path):
		with open(json_path) as f:
			self.__dict__.update(json.load(f))
		
	@property
	def dict(self):
		return self.__dict__


