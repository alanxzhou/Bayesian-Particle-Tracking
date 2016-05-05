from setuptools import setup

setup(name='bayesian-particle-tracking',
      version='0.1',
      description='Bayesian Particle Tracking',
      url='http://github.com/alanzhou93/bayesian-particle-tracking',
      author='Alan Zhou',
      author_email='alanzhou@college.harvard.edu',
      license='General Public License',
      packages=['Bayesian_Particle_Tracking'],
      install_requires=['numpy', 'matplotlib', 'scipy'],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)