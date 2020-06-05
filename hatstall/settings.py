from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline as ModelPipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler

from hatstall.pipes import Pipeline as CustomPipeline, PipelineSystem
from hatstall.pipes.engineer import PostsJoiner, AverageWordCalculator
from hatstall.pipes.loader import PersonalityPostLoader
from hatstall.pipes.model import Evaluator
from hatstall.pipes.preparator import (
    PostsSplitter, EvenlyDistributor, TrainTestSplitter, DigitReplacer,
    LinkReplacer, PersonalityCodeReplacer)


PreparationPipeline = CustomPipeline
EvaluationPipeline = CustomPipeline

PIPELINE_SYSTEM = PipelineSystem([
    ('preperation', PreparationPipeline([
        ('loader', PersonalityPostLoader),
        ('splitter', PostsSplitter),
        ('persona', PersonalityCodeReplacer),
        ('link', LinkReplacer),
        ('digit', DigitReplacer),
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
    ]))
])
