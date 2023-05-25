from run import run
from utils import get_joint_config
import wandb


if __name__ == "__main__":
    wandb.init()
    cfg = get_joint_config(wandb.config)
    wandb.config.update({'name': cfg['experiment_name']})
    run(cfg, report_to='wandb')
