from .nn import save_model
from ..queries.visualize import render_sdf_2d, reconstruct_surface_marching_cubes
import mouette as M
import os
import csv

class Callback:
    """
    An empty Callback object to be called inside a Trainer (see trainers/base.py)

    Callback affect the trainer they are associated with, or provide log infos, or anything you can think of.
    Inside a Trainer, they can be called at three points:
    - At the beginning of an training epoch
    - At the end of an training epoch
    - At the end of a forward/backward pass
    - At the end of a testing epoch
    - At the very end of the training procedure
    """
    def callOnEndForward(self, trainer, model):
        pass

    def callOnBeginTrain(self, trainer, model):
        pass
    
    def callOnEndTrain(self, trainer, model):
        pass

    def callOnEndTest(self, trainer, model):
        pass
    

class LoggerCB(Callback):

    def __init__(self, file_path: str):
        """Registers metrics of the training inside a .log file

        Args:
            file_path (str): path to the log text file.
        """
        super().__init__()
        self.path = file_path
        self.logged = {"epoch" : -1, "time" : 0, "train_loss" : -1, "test_loss" : -1}
        with open(self.path, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.logged.keys())
            writer.writeheader()

    def callOnEndTrain(self, trainer, model):
        self.logged.update({"epoch" :  trainer.metrics["epoch"]})
        self.logged.update({"time" :  trainer.metrics["epoch_time"]})
        self.logged.update({"train_loss" : trainer.metrics["train_loss"]})
        print(f"Train loss after epoch {trainer.metrics['epoch']} : {trainer.metrics['train_loss']}")
        if trainer.test_data_loader is None:
            # no test_data_loader means that callOnEndTest will not be called.
            self._write_log()

    def callOnEndTest(self, trainer, model):
        self.logged.update({"test_loss" : trainer.metrics["test_loss"]})
        print(f"Test loss after epoch {trainer.metrics['epoch']} : {trainer.metrics['test_loss']}\n")
        self._write_log()

    def _write_log(self):
        with open(self.path, "a") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.logged.keys())
            writer.writerow(self.logged)

class CheckpointCB(Callback):

    def __init__(self, save_folder: str, when: list):
        """A Callback responsible for saving the model currently in training into a file

        Args:
            save_folder (str): folder into which the model will be saved. The filename if formatted as `model_e{epoch}.pt`
            when (list): list of epochs when the model should be saved onto dist
        """
        self.save_folder: str = save_folder
        self.when = when

    def callOnEndTrain(self, trainer, model):
        epoch = trainer.metrics["epoch"]
        if epoch>0 and epoch in self.when:
            name = f"model_e{epoch}.pt"
            path = os.path.join(self.save_folder, name)
            save_model(model, path)

class Render2DCB(Callback):

    def __init__(self, save_folder: str, freq: int, plot_domain: M.geometry.AABB = None, resolution: int = 800, output_contours: bool = True, output_gradient_norm: bool = True):
        """A Callback that makes a snapshot of a 2D neural implicit by sampling its values on a grid. Can also sample the gradient's norm and make a contour plot.

        Args:
            save_folder (str): output folder into which the images are saved
            freq (int): frequency (in terms of number of epochs) at which a snapshot is taken
            plot_domain (M.geometry.AABB, optional): Spanning domain of the taken snapshot.  If not provided, the domain will be taken as a default [-1.5, 1.5]^2. Defaults to None.
            resolution (int, optional): Resolution of the snapshot grid. resolution^2 samples will be computed from the neural implicit model. Defaults to 800.
            output_contours (bool, optional): Whether to output a contour plot of the neural field. Defaults to True.
            output_gradient_norm (bool, optional): Whether to also output a plot of the norm of the neural field's gradient. Defaults to True.

        Warning:
            Fails if the neural implicit currently training is not 2-dimensionnal.
        """
        super().__init__()
        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.freq = freq
        if plot_domain is None:
            self.domain = M.geometry.AABB([-1.5,-1.5],[1.5,1.5])
        else:
            self.domain = plot_domain
        self.freq = freq
        self.res = resolution
        self.output_contours = output_contours
        self.output_gradient_norm = output_gradient_norm

    def callOnEndTrain(self, trainer, model):
        epoch = trainer.metrics["epoch"]
        if self.freq>0 and epoch%self.freq==0:
            render_path = os.path.join(self.save_folder, f"render_{epoch}.png")
            contour_path = os.path.join(self.save_folder, f"contour_{epoch}.png") if self.output_contours else None
            gradient_path = os.path.join(self.save_folder, f"grad_{epoch}.png") if self.output_gradient_norm else None
            render_sdf_2d(
                render_path,
                contour_path,
                gradient_path,
                model, 
                self.domain, 
                trainer.config.DEVICE, 
                res=self.res, 
                batch_size=trainer.config.TEST_BATCH_SIZE,
            )

class MarchingCubeCB(Callback):
    def __init__(self, save_folder: str, freq: int, domain: M.geometry.AABB = None, res: int = 100, iso=0):
        """A Callback that makes a snapshot of a 3D neural implicit by using the marching cubes algorithm to extract some level sets.

        Args:
            save_folder (str): output folder into which the images are saved
            freq (int): frequency (in terms of number of epochs) at which a snapshot is taken
            domain (M.geometry.AABB, optional): AABB domain over which the grid is defined. If not provided, the default domain will be [-1.2 ; 1.2]^3. Defaults to None.
            res (int, optional): Grid resolution for marching cubes. res^3 values will be sampled from the neural model. Defaults to 100.
            iso (int, optional): Which iso-level will be reconstructed. Several levels can be provided in a list. Defaults to 0.
        """
        super().__init__()
        self.save_folder = save_folder
        self.freq = freq
        if domain is None:
            self.domain = M.geometry.AABB([-1.2]*3, [1.2]*3)
        else:
            self.domain = domain
        self.res = res
        if isinstance(iso,float):
            self.iso = [iso]
        else:
            self.iso = iso
    
    def callOnEndTrain(self, trainer, model):
        epoch = trainer.metrics["epoch"]
        if self.freq>0 and epoch%self.freq==0:
            try:
                iso_surfaces = reconstruct_surface_marching_cubes(
                    model, 
                    self.domain, 
                    trainer.config.DEVICE, 
                    self.iso, 
                    self.res, 
                    trainer.config.TEST_BATCH_SIZE)
                for (n,off),mesh in iso_surfaces.items():
                    M.mesh.save(mesh, os.path.join(self.save_folder, f"e{epoch:04d}_n{n:02d}_iso{round(1000*off)}.obj"))
            except Exception as e:
                print("[ERROR] Marching Cube Callback:", e)
                pass