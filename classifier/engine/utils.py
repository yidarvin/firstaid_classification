

class ClassifierOutputTarget_modified:
    def __init__(self, task, category):
        self.task = task
        self.category = category
    def __call__(self, model_output):
        print(model_output)
        if len(model_output[0]) == 1:
            return model_output[self.task][self.category]
        return model_output[self.task][:, self.category]
