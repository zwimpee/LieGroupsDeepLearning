import os
from tfx.components import CsvExampleGen, TensorBoard, Pusher
from tfx.components.trainer.component import TrainerComponent
from tfx.orchestration import pipeline
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from tfx.proto import trainer_pb2

# Set the paths and directories
_pipeline_name = 'my_pipeline'
_pipeline_root = os.path.join('pipeline', _pipeline_name)
_data_path = 'C:\Users\User\LieGroupsDeepLearning\Chapter 1 - Introduction to Lie Groups\Code\wikitext-2'
_log_path = 'C:\Users\User\LieGroupsDeepLearning\Chapter 1 - Introduction to Lie Groups\Notes\log.txt'

# Define the data ingestion component
example_gen = CsvExampleGen(input_base=_data_path)

# Define the trainer component
trainer = TrainerComponent(
    trainer_pb2.TrainArgs(num_steps=1000),
    trainer_pb2.EvalArgs(num_steps=100),
    module_file='/Users/User/LieGroupsDeepLearning/Chapter 1 - Introduction to Lie Groups/Code/train.py',  # Path to your existing training script
    transformed_examples=example_gen.outputs['examples'])

# Define the TensorBoard component
tensorboard = TensorBoard(
    log_dir=_log_path,
    enable_cache=True)  # Enable caching for TensorBoard

# Define the pipeline
components = [example_gen, trainer, tensorboard]
pipeline_name = 'test_pipeline'
p = pipeline.Pipeline(
    pipeline_name=pipeline_name,
    pipeline_root=_pipeline_root,
    components=components)

# Run the pipeline
context = InteractiveContext()
context.run(p)
