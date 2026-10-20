import torch
from torch import nn

from functools import wraps
import inspect
import importlib


class ImplicitModel(nn.Module):
    """Base class of all neural models of the library.

    Provides the utilities for saving/loading models onto/from the disk. Any subclass inheriting this one will automatically store the arguments given to its constructor, so that they can be saved alongside the model's parameters and reloaded in another session.

    ```python
    class ExampleNet(ImplicitModel):
            def __init__(self, x:int = 2):
                super().__init__()
                self.x = x
    
        ExampleNet(3).save("example_model.inr")
        model = implicitlab.nn.load_model("example_model.inr") # an ExampleNet instance
        print(model.x)
    ```
    """
    def __init__(self):
        super().__init__()

    def __init_subclass__(cls):
        """
        This code is called every time a children class is defined. Will wrap the __init__ function of the children class so that its arguments are stored in the `_config` attribute.
        """
        if not "__init__" in cls.__dict__: return
        init_fun = cls.__dict__["__init__"]
        signature = inspect.signature(init_fun)
        self_name = next(iter(signature.parameters))
    
        @wraps(init_fun)
        def wrapped_init(self, *args, **kwargs):
            # Only the outermost constructor records its arguments, so that calls to super().__init__ in subclasses do not override them
            if "_config" not in self.__dict__:
                bound = signature.bind(self, *args, **kwargs)
                bound.apply_defaults()
                config = dict(bound.arguments)
                config.pop(self_name)
                object.__setattr__(self, "_config", config)
            init_fun(self, *args, **kwargs)
        cls.__init__ = wrapped_init


    def save(self, path: str):
        """Saves the model to the disk in a file prescribed by `path`. This methods saves both the model's weights as well as its configuration, i.e. its type as a subclass of ImplicitModel and the arguments of its constructor. That way, the model can be re-instanciated automatically provided.

        Args:
            path (str): path to the file into which the model will be saved. We use `.inr` extension as a convention.
        """
        to_save = {
            "file_type" : "IMPLICIT_LAB_MODEL",
            "config" : self._config, 
            "path" : f"{self.__module__}:{self.__class__.__qualname__}",
            "weights" : self.state_dict()
        }
        torch.save(to_save, path)
    
    @classmethod
    def load(cls, path:str, map_location:str = "cpu"):
        """Loads an implicit model from a file on the disk.

        Args:
            path (str): path to the saved file.
            map_location (str, optional): `map_location` parameter for `torch.load`. Defaults to "cpu".
        
        Raises:
            Exception: Fails if the loaded file is not an implicitlab model file.

        Returns:
            ImplicitModel: the loaded model with correct type, architecture and weights.
        """
        data = torch.load(path, map_location=map_location, weights_only=False)
        if data["file_type"] != "IMPLICIT_LAB_MODEL":
            raise Exception("The loaded file is recognized as an implicitlab model.")
        config, path = data["config"], data["path"] # find the corresponding path

        # Find the exact class to call
        module_name, qualname = path.split(":")
        subclass = importlib.import_module(module_name)
        for attr in qualname.split("."):
            subclass = getattr(subclass, attr)

        # Transform the config dict into arguments for the class constructor
        signature = inspect.signature(subclass.__init__)
        # Determines which arguments are positional or keyword
        args, kwargs = [], {}
        has_var_positional : bool = any(p.kind == p.VAR_POSITIONAL for p in signature.parameters.values())
        before_var_positional : bool = True
        for name, param in signature.parameters.items():
            if name not in config: continue
            if param.kind == param.VAR_POSITIONAL:
                # something like f(*x) is stored in config["x"] as an iterable
                args.extend(config[name])
                before_var_positional = False
            elif param.kind == param.POSITIONAL_ONLY or (has_var_positional and before_var_positional):
                args.append(config[name])
            elif param.kind == param.VAR_KEYWORD:
                kwargs.update(config[name])
            else:
                kwargs[name] = config[name]
        model = subclass(*args, **kwargs)
        model.load_state_dict(data["weights"])
        return model


def load_model(path: str, map_location: str = "cpu") -> ImplicitModel:
    """Loads an implicit model. Alias for `ImplicitModel.load`

    ```python
    # The two lines below are equivalent:
    model = IL.load_model("path/to/model.inr")
    model = IL.nn.ImplicitModel.load("path/to/model.inr")
    ```
    """
    return ImplicitModel.load(path, map_location)


class ImplicitSequential(ImplicitModel):
    """Equivalent of `torch.nn.Sequential` but with the implicitlab's save/load feature.
    Chains several models, for instance an input encoding followed by a network:

    ```python
    model = ImplicitSequential(
        RandomFourierEncoding(3, 256), 
        SirenNet(256, 128, 4)
    )
    ```
    """
    def __init__(self, *models: nn.Module):
        """
        Args:
            *models (nn.Module): models being part of the sequence.
        """
        super().__init__()
        self.seq_models = nn.Sequential(*models)

    def forward(self, x):
        return self.seq_models(x)

    def __getitem__(self, i):
        return self.seq_models[i]

    def __len__(self):
        return len(self.seq_models)

    def __iter__(self):
        return iter(self.seq_models)