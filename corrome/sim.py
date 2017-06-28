
def chain_interactions(gradient, mus, sigmas, rng=None):
    """
    This generates an urn simulating a chain of interacting species.
    This commonly occurs in the context of a redox tower, where
    multiple species are distributed across a gradient.

    Parameters
    ----------
    gradient: array_like
       Vector of values associated with an underlying gradient.
    mus: array_like
       Vector of means.
    sigmas: array_like
       Vector of standard deviations.
    rng: np.random.RandomState
       Numpy random state.

    Returns
    -------
    np.array
       A matrix of real-valued positive abundances where
       there are `n` rows and `m` columns where `n` corresponds
       to the number of samples along the `gradient` and `m`
       corresponds to the number of species in `mus`.
    """
    pass

def multinomial_sample(X, lam, rng=None):
    """
    This draws multinomial samples from an urn using some poisson
    process denoted by lam.

    Parameters
    ----------
    X: array_like
       A matrix of real-valued positive abundances where
       there are `n` rows and `m` columns where `n` corresponds
       to the number of samples and `m` corresponds to the number
       of species.
    n: array_like
       Sequencing depth for each
    rng: np.random.RandomState
       Numpy random state.

    Returns
    -------
    np.array:
       A matrix of counts where
       there are `n` rows and `m` columns where `n` corresponds
       to the number of samples and `m` corresponds to the number
       of species.
    """
    pass

def compositional_noise(sigma, nsamp, rng=None):
    """
    This is multiplicative noise applied across the entire dataset.

    Parameters
    ----------
    rng: np.random.RandomState
       Numpy random state.
    sigma: array_like
       Vector of standard deviations in ilr space.
    nsamp: int
       Number of samples to generate

    Returns
    -------
    np.array:
       A matrix of probabilities where there are `n` rows and
       `m` columns where `n` corresponds to the number of samples
       and `m` corresponds to the number of species.

    """
    pass

def count_noise(lam, p, nsamp, rng=None):
    """
    This is count noise applied across the entire dataset.
    Specifically, it is a multinomial poisson process, where the
    multinomial represents the contamination probabilities and the
    poission represents the sequencing depth.

    Parameters
    ----------
    lam : float
       Poisson parameter, which is also the mean and variance
       of the Poisson.
    p : array, float
       Vector probabilities associated with the multinomial.
    nsamp: int
       Number of samples to generate
    rng: np.random.RandomState
       Numpy random state.

    Returns
    -------
    np.array
       A matrix of counts where there are `n` rows and `m` columns
       where `n` corresponds to the number of samples and `m`
       corresponds to the number of species.

    """
    pass

def train_count_parameters(data):
    """
    Given a noisy data, try to learn the count noise parameters.
    This assumes that there is only a single underlying urn.
    So the multinomial probabilties are just an aggregrate of all
    of the counts.

    Parameters
    ----------
    data : array_like
       A matrix of counts where there are `n` rows and `m` columns
       where `n` corresponds to the number of samples and `m`
       corresponds to the number of species.

    Returns
    -------
    lam: float
       Poisson parameter for generating sequencing depths.
    p: np.array
       Vector of multinomial probabilities.
    """
    pass

def train_compositional_parameters(data):
    """
    Given noisy compositional data, try to learn the compositional noise
    parameters.


    Parameters
    ----------
    data : array_like
       A matrix of counts where there are `n` rows and `m` columns
       where `n` corresponds to the number of samples and `m`
       corresponds to the number of species.

    Returns
    -------
    mu: float
       Mean of ilr normal.
    sigma: float
       Standard deviation of ilr normal.
    """
    pass


