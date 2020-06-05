from collections import Counter
import itertools
import random
import re

from sklearn.model_selection import train_test_split

from hatstall.pipes import Pipe
from hatstall.pipes.loader import Person, Personality


class PersonContainer:
    def __init__(self, persons):
        self.persons = persons

    def get_personality(self):
        return [x.personality for x in self.persons]

    def get_posts(self):
        return [x.posts for x in self.persons]

    def evenly_distribute(self):
        first = list(filter(
            lambda x: x.personality == Personality.FIRST_PERSONALITY,
            self.persons))
        second = list(filter(
            lambda x: x.personality == Personality.SECOND_PERSONALITY,
            self.persons))
        if len(first) > len(second):
            first_small = first[:len(second)]
            self.persons = second + first_small
        elif len(first) < len(second):
            second_small = second[:len(first)]
            self.persons = first + second_small
        else:
            print("Both personalities already have an even distribution.")
        random.shuffle(self.persons)

    def _chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def split_posts(self, chunk_size):
        new_persons = []
        ignored_posts = 0
        for person in self.persons:
            posts = person.posts
            personality = person.personality
            for chunk in self._chunks(posts, chunk_size):
                if len(chunk) == chunk_size:
                    new_persons.append(Person(personality, chunk))
                else:
                    ignored_posts += len(chunk)
        if ignored_posts != 0:
            print(
                "Ignored %d texts because of too small chunks" % (
                    ignored_posts))
        self.persons = new_persons

    def replace_digits(self):
        self._replace_patterns_in_post(r'(\S*\d\S*)', '$digit')

    def replace_links(self):
        self._replace_patterns_in_post(
            r'(\S*www\.[a-z|\-]{2,}\.[a-z]{2,}\S*)', '$link')

    def replace_personality_codes(self):
        personalities = [('i', 'e'), ('n', 's'), ('t', 'f'), ('j', 'p')]
        personalities = [
            ''.join(x) for x in list(itertools.product(*personalities))]
        self._replace_patterns_in_post(
            r'\S*%s\S*' % '\S*|\S*'.join(personalities), '')

    def _replace_patterns_in_post(self, pattern, replacer):
        for person in self.persons:
            person.posts = [re.sub(pattern, replacer, x) for x in person.posts]


# TODO check if container exists if not create one. if it does load from
# payload. right now we have to use the PostsSplitter pipe to generate the
# container which is used later in the other pipes
class PostsSplitter(Pipe):

    CHUNK_SIZE = 10

    def run(self):
        persons = self.payload['persons']
        container = PersonContainer(persons)
        container.split_posts(self.CHUNK_SIZE)
        self.payload['persons_container'] = container
        self.payload['persons'] = container.persons


class EvenlyDistributor(Pipe):

    def run(self):
        container = self.payload['persons_container']
        container.evenly_distribute()
        self.payload['persons'] = container.persons


class DigitReplacer(Pipe):

    def run(self):
        container = self.payload['persons_container']
        container.replace_digits()
        self.payload['persons'] = container.persons


class LinkReplacer(Pipe):

    def run(self):
        container = self.payload['persons_container']
        container.replace_links()
        self.payload['persons'] = container.persons


class PersonalityCodeReplacer(Pipe):

    def run(self):
        container = self.payload['persons_container']
        container.replace_personality_codes()
        self.payload['persons'] = container.persons


class TrainTestSplitter(Pipe):

    def run(self):
        data = self.payload['persons_container']
        train, test = train_test_split(
            data.persons, test_size=0.3,
            random_state=123,
            stratify=data.get_personality())

        train_cont = PersonContainer(train)
        print(f"Train size: {Counter(train_cont.get_personality())}")

        test_cont = PersonContainer(test)
        print(f"Test size: {Counter(test_cont.get_personality())}")

        train_x = train_cont.get_posts()
        train_y = train_cont.get_personality()

        test_x = test_cont.get_posts()
        test_y = test_cont.get_personality()
        self.payload['train_test'] = (train_x, train_y, test_x, test_y)
