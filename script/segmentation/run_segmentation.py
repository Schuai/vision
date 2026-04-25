import click

from vision.cfg.segmentation.seg_cfg import SegmentationCfg
from vision.wrapper.seg_wrapper import SegWrapper


@click.command()
@click.option(
    "--seg-cfg",
    default="config/tracking/seg_cfg_etam.yaml",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    help="Path to a segmentation YAML config.",
)
@click.option("--device", default="cuda", show_default=True, help="Torch device for the predictor.")
def main(seg_cfg: str, device: str) -> None:
    """Load a segmentation camera predictor."""
    cfg = SegmentationCfg.from_yaml(seg_cfg)
    wrapper = SegWrapper(cfg=cfg, device=device)
    click.echo(
        f"Loaded {wrapper.model_type} predictor with config "
        f"{wrapper.seg_config} and checkpoint {wrapper.seg_checkpoint}"
    )


if __name__ == "__main__":
    main()
