from setuptools import setup


setup(\
        name="BootstrapEvo",\
        py_modules=["bevodevo"],\
        install_requires=["numpy==1.18.4",\
                        "gym[box2d,classic_control]~=0.15.3",\
                        "scikit-image==0.17.2",\
                        "pybullet==3.0.7",\
                        "matplotlib==3.1.2"],\
        version="0.0",\
        description="BevoDevo: Bootstrapping Deep Neuroevolution and Neurodevelopment",\
        author="Rive Sunder",\
        )
