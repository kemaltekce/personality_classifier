import csv

from hatstall.pipes import Pipe


class Personality:
    PERSONALITY_IDX = 0
    FIRST_PERSONALITY = 'Introvert'
    SECOND_PERSONALITY = 'Extrovert'


class Person:
    def __init__(self, personality, posts):
        self.personality = self._assign_personality(personality)
        self.posts = self._assign_posts(posts)

    def _assign_personality(self, personality):
        persona = personality[Personality.PERSONALITY_IDX]
        frist_p = Personality.FIRST_PERSONALITY
        second_p = Personality.SECOND_PERSONALITY
        message = (
            f'Personality of {frist_p[Personality.PERSONALITY_IDX]} '
            f'({frist_p}) or {second_p[Personality.PERSONALITY_IDX]} '
            f'({second_p}) is expected.')
        assert persona in (
            frist_p[Personality.PERSONALITY_IDX],
            second_p[Personality.PERSONALITY_IDX]), message
        if persona == frist_p[Personality.PERSONALITY_IDX]:
            return frist_p
        elif persona == second_p[Personality.PERSONALITY_IDX]:
            return second_p

    def _assign_posts(self, posts):
        # inside the raw data the different posts are divided by |||
        if '|||' in posts:
            return posts.split('|||')
        elif isinstance(posts, list):
            return posts
        else:
            raise ValueError("Can't handle the posts input.")


class PersonalityPostLoader(Pipe):

    def run(self):
        with open('../data/mbti_1.csv') as file:
            data = csv.reader(file, delimiter=',')
            # ignore header
            next(data, None)
            persons = []
            for row in data:
                if '|||' in row[1][1:-1]:
                    persons.append(Person(row[0], row[1][1:-1].lower()))
        self.payload['persons'] = persons
