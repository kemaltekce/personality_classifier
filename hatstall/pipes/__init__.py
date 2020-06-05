class Pipe:
    def __init__(self, payload, nickname=None):
        self.payload = payload
        self.nickname = nickname

    def run(self):
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__


class PipelineContainer:
    def __init__(self, pipeline, nickname=None):
        self.pipeline = pipeline
        self.nickname = nickname


class Pipeline:
    def __init__(self, pipes):
        self.payload = {}
        self.input_pipes = pipes
        self.pipes = self._initialize_pipes(pipes)

    def _initialize_pipes(self, pipes):
        pipes = [pipe(self.payload, nickname) for nickname, pipe in pipes]
        return pipes

    def run(self):
        for pipe in self.pipes:
            print("Running %s pipe --> %s" % (pipe.nickname, pipe.name))
            pipe.run()

    def reinitialize_pipes(self):
        self.pipes = [
            pipe(self.payload, nickname) for nickname, pipe in
            self.input_pipes]


class PipelineSystem:
    def __init__(self, pipelines):
        self.pipeline_containers = self._load_pipelines(pipelines)

    def _load_pipelines(self, pipelines):
        pipelines = [
            PipelineContainer(pipeline, nickname) for nickname, pipeline
            in pipelines]
        return pipelines

    def run(self):
        for pl_container in self.pipeline_containers:
            if pl_container.nickname == 'preperation':
                prep_pipeline = pl_container.pipeline
                print("--- Running preparation pipeline ---")
                prep_pipeline.run()
            elif pl_container.nickname == 'modelling':
                print("--- Running modelling pipeline ---")
                train_x, train_y, _, _ = (
                    prep_pipeline.payload['train_test'])
                model_pipeline = pl_container.pipeline
                model_pipeline.fit(train_x, train_y)
                prep_pipeline.payload['model'] = model_pipeline
            elif pl_container.nickname == 'evaluation':
                print("--- Running evaluation pipeline ---")
                pl_container.pipeline.payload = prep_pipeline.payload
                pl_container.pipeline.reinitialize_pipes()
                pl_container.pipeline.run()
            else:
                raise ValueError(
                    "Don't recognize pipeline nickname: %s" %
                    pl_container.nickname)
