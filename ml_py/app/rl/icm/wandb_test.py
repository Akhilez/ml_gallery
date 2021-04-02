import wandb
from settings import BASE_DIR


def my_train_loop():
    for epoch in range(10):
        loss = 0  # change as appropriate :)
        wandb.log(
            {
                "epoch": epoch,
                "loss": loss,
                "rewards": wandb.Histogram([1, 4, 3, 5, 6, 3, 2]),
            }
        )


def main():

    wandb.init(
        # name="",  # Name of the run
        project="sample-project",
        save_code=True,
        group=None,
        tags=None,  # List of string tags
        notes=None,  # longer description of run
        dir=BASE_DIR,
    )
    cfg = wandb.config
    cfg.post_init = True
    cfg.learning_rate = 0.001

    my_train_loop()


if __name__ == "__main__":

    main()
