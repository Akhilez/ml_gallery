import wandb
from settings import BASE_DIR


def main():

    run = wandb.init(
        # name="",  # Name of the run
        project="sample-project",
        config={},
        save_code=True,
        group=None,
        tags=None,  # List of string tags
        notes=None,  # longer description of run
        dir=BASE_DIR,
    )
    cfg = run.config
    cfg.post_init = True
    cfg.learning_rate = 0.001

    for epoch in range(10):
        loss = 0  # change as appropriate :)
        run.log({"epoch": epoch, "loss": loss})


if __name__ == "__main__":
    main()
