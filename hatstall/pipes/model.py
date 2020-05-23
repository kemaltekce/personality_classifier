from sklearn.metrics import confusion_matrix, f1_score

from hatstall.pipes import Pipe
from hatstall.pipes.loader import Personality


class Evaluator(Pipe):

    def run(self):
        model = self.payload['model']
        _, _, test_x, test_y = self.payload['train_test']
        print(model.score(test_x, test_y))
        print(f1_score(
            test_y, model.predict(test_x), average=None,
            labels=[
                Personality.FIRST_PERSONALITY,
                Personality.SECOND_PERSONALITY
            ]))
        cm = confusion_matrix(
            test_y, model.predict(test_x),
            labels=[
                Personality.FIRST_PERSONALITY,
                Personality.SECOND_PERSONALITY
            ])
        print(cm)
