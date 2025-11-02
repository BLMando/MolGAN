import tensorflow as tf

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def wasserstein_loss(real_scores, fake_scores):
    """
    Wasserstein loss for GAN
    """
    return tf.reduce_mean(fake_scores) - tf.reduce_mean(real_scores)


def gradient_penalty(discriminator, real_data, fake_data):
    """
    Compute gradient penalty for WGAN-GP
    
    Parameters:
    -----------
    discriminator: ProcessTraceDiscriminator
        Discriminator model
    real_data: tf.Tensor [batch_size, max_length, num_activities]
        Real traces (one-hot encoded)
    fake_data: tf.Tensor [batch_size, max_length, num_activities]
        Fake traces (soft probabilities)
    
    Returns:
    --------
    penalty: tf.Tensor scalar
        Gradient penalty value
    """
    batch_size = tf.shape(real_data)[0]

    # Random interpolation coefficient
    alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)

    # Interpolate between real and fake data
    interpolated = alpha * real_data + (1 - alpha) * fake_data

    # Compute discriminator output for interpolated data
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        scores = discriminator(interpolated, is_discrete=False, training=True)

    # Compute gradients
    gradients = tape.gradient(scores, interpolated)

    # Compute gradient penalty
    gradients = tf.reshape(gradients, [batch_size, -1])
    gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
    penalty = tf.reduce_mean(tf.square(gradient_norm - 1.0))

    return penalty


# ============================================================================
# GUMBEL-SOFTMAX UTILITIES
# ============================================================================

def sample_gumbel(shape, eps=1e-20):
    """
    Sample from Gumbel(0, 1) distribution
    """
    U = tf.random.uniform(shape, minval=0, maxval=1)
    return -tf.math.log(-tf.math.log(U + eps) + eps)


def gumbel_softmax(logits, temperature, hard=False):
    """
    Sample from the Gumbel-Softmax distribution
    
    Parameters:
    -----------
    logits: tf.Tensor [batch_size, num_classes]
        Unnormalized log probabilities
    temperature: float
        Temperature parameter (higher = more uniform)
    hard: bool
        If True, return one-hot (with straight-through gradient)
        If False, return soft probabilities
    
    Returns:
    --------
    y: tf.Tensor [batch_size, num_classes]
        Sample from Gumbel-Softmax distribution
    """
    gumbel_noise = sample_gumbel(tf.shape(logits))
    y = logits + gumbel_noise
    y = tf.nn.softmax(y / temperature)

    if hard:
        # Straight-through estimator
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(
            y, axis=-1, keepdims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y

    return y
