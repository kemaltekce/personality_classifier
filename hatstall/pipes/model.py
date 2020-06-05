import numpy as np
import re

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


class TestDataTokenChecker(Pipe):

    def run(self):
        # get feature names of vectorizer/tokenizer
        model = self.payload['model']
        feature_pipeline = model.named_steps['features']
        token_pipeline = [
            x for x in feature_pipeline.transformer_list
            if x[0] == 'token'][0][1]
        tokens = token_pipeline.named_steps['vect'].get_feature_names()

        # get token pattern
        pattern = token_pipeline.named_steps['vect'].token_pattern

        # only keep data for which models knows more than 70% of the words
        train_x, train_y, test_x, test_y = self.payload['train_test']
        test_x_ = [' '.join(x) for x in test_x]
        test_tokens_x = [re.findall(pattern, x) for x in test_x_]
        missing = np.array([
            len((set(x) - set(tokens))) / len(set(x)) for x in test_tokens_x])
        keep = missing < 0.3
        print(
            f"Removing {len(test_x) - sum(keep)} out of {len(test_x)} test "
            "examples because model knows to few words in test example.")

        test_x = np.array(test_x)
        test_x = list(test_x[keep])
        test_y = np.array(test_y)
        test_y = list(test_y[keep])
        self.payload['train_test'] = train_x, train_y, test_x, test_y


class Predictor(Pipe):

    def run(self):
        model = self.payload['model']
        data = self.payload['persons_container']
        X = data.get_posts()
        pred = model.predict(X)

        names = self.payload['names']
        print('--- Here are the predictions ---')
        for i, name in enumerate(names):
            print(f"{name} is an {pred[i]}")
