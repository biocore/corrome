
def chain_interactions(mus, sigmas):
    """
    This generates an urn simulating a chain of interacting species.
    This commonly occurs in the context of a redox tower, where
    multiple species are distributed across a gradient.
    """
    pass

def multinomial_sample(X):
    """
    This draws multinomial samples from an urn.
    """
    pass

def compositional_noise(sigma):
    """
    This is multiplicative noise applied across the entire dataset.
    """
    pass

def count_noise(lam, p):
    """
    This is count noise applied across the entire dataset.
    Specifically, it is a multinomial poisson process, where the
    multinomial represents the contamination probabilities and the
    poission represents the sequencing depth.
    """
    pass

def train_count_parameters(data):
    """
    Given a noisy data, try to learn the count noise parameters.
    """
    pass

def train_compositional_parameters(data):
    """
    Given noisy compositional data, try to learn the compositional noise
    parameters.
    """
    pass


