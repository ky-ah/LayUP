import wandb


class LoggerBackend:
    def __init__(self, **kwargs):
        pass

    def log(self, log_dict):
        pass

    def close(self):
        pass


class ConsoleLogger(LoggerBackend):
    def log(self, log_dict):
        for k, v in log_dict.items():
            print(f"{k} => {v}")


class TQDMLogger(LoggerBackend):
    def __init__(self, pbar, postfix=True, **kwargs):
        super().__init__(**kwargs)
        self.postfix = postfix
        self.pbar = pbar

    def log(self, log_dict):
        if self.postfix:
            self.pbar.set_postfix(log_dict)
        else:
            self.pbar.set_description(log_dict)


class WandbLogger(LoggerBackend):
    def __init__(
        self, wandb_project, wandb_entity, args, wandb_run_name=None, **kwargs
    ):
        # check if args is dict and if not, convert it
        if not isinstance(args, dict):
            args = vars(args)

        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            config=args,
        )

    def log(self, log_dict):
        wandb.log(log_dict)

    def close(self):
        wandb.finish()
