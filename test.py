from utils import display_pipeline, align_and_display
from load import load_bag_file, load_live_stream

# Test feed
pipeline, config = load_live_stream()
#display_pipeline(pipeline, config)
align_and_display(pipeline, config)

# Test recording
filename = "data/20231201_152820.bag"
pipeline, config = load_bag_file(filename)
align_and_display(pipeline, config)
