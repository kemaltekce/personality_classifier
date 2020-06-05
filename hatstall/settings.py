from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline as ModelPipeline
from sklearn.preprocessing import MinMaxScaler

from hatstall.pipes import Pipeline as CustomPipeline, PipelineSystem
from hatstall.pipes.engineer import AverageWordCalculator, PostsJoiner
from hatstall.pipes.loader import PersonalityPostLoader, PredictionDataLoader
from hatstall.pipes.model import Evaluator, Predictor
from hatstall.pipes.preparator import (
    DigitReplacer, EvenlyDistributor, LinkReplacer, PostsSplitter,
    PersonalityCodeReplacer, TrainTestSplitter)


PreparationPipeline = CustomPipeline
EvaluationPipeline = CustomPipeline
PredictionPipeline = CustomPipeline

GLOBEL_PREP_PIPES = [
        ('persona', PersonalityCodeReplacer),
        ('link', LinkReplacer),
        ('digit', DigitReplacer),
]

PIPELINE_SYSTEM = PipelineSystem([
    ('preperation', PreparationPipeline([
        ('loader', PersonalityPostLoader),
        ('splitter', PostsSplitter),
        *GLOBEL_PREP_PIPES,
        ('evenfier', EvenlyDistributor),
        ('traintest', TrainTestSplitter)
    ])),
    ('modelling', ModelPipeline([
        ('features', FeatureUnion([
            ('token', ModelPipeline([
                ('joiner', PostsJoiner()),
                ('vect', TfidfVectorizer(
                    max_df=1.0, min_df=0.01,
                    token_pattern='(?u)\\$?\\b\\w\\w+\\b')),
            ])),
            ('mean_word', ModelPipeline([
                ('average', AverageWordCalculator()),
                ('scaler', MinMaxScaler())
            ])),
        ])),
        ('log', LogisticRegression(random_state=123)),
    ])),
    ('evaluation', EvaluationPipeline([
        ('eval', Evaluator)
    ])),
    ('prediction', PredictionPipeline([
        ('loader', PredictionDataLoader),
        *GLOBEL_PREP_PIPES,
        ('predictor', Predictor),
    ])),
])
