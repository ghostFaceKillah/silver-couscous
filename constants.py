PLAYBACK_MEMORY_SIZE = 10000
TRAINING_BATCH_SIZE = 32

ORIGINAL_IMAGE_H = 210
ORIGINAL_IMAGE_W = 160

RESIZED_IMAGE_H = 84
RESIZED_IMAGE_W = 84

NUM_FRAMES_PASSED = 3
ACTION_SPACE_SIZE = 6

STARTING_EPSILON = 1.0
EPSILON_DECAY_LENGTH = 1000000
MIN_EPSILON = 0.1
EPSILON_DECAY = (STARTING_EPSILON - MIN_EPSILON) / EPSILON_DECAY_LENGTH

# TODO(mike): Check this param
DISCOUNT_FACTOR = 0.99

RESET_TARGET_NET_FREQUENCY = 10000

SAVE_FREQ = 10000
