"""Argparse flags generated from a config dataclass (shared by the IPPO and BC trainers).

ManiSkill's PPO baseline uses `tyro` for this; it is not a dependency of this project, so every
dataclass field gets one flag (`--num-envs`, `--no-anneal-lr`, ...). Stdlib only.
"""

import argparse
import dataclasses
import types
import typing
from collections.abc import Callable, Mapping, Sequence


def optional_of(parse: Callable[[str], object]) -> Callable[[str], object]:
    """Argparse type that also accepts `none` / `null` for an `X | None` field."""

    def convert(text: str) -> object:
        return None if text.lower() in ("none", "null") else parse(text)

    convert.__name__ = parse.__name__
    return convert


def unwrap_optional(annotation: object) -> tuple[type, bool]:
    """`(X, True)` for `X | None`, `(X, False)` for a plain `X`."""
    if isinstance(annotation, types.UnionType) or typing.get_origin(annotation) is typing.Union:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return annotation, False  # type: ignore[return-value]


def build_dataclass_parser(
    config_cls: type,
    description: str,
    choices: Mapping[str, Sequence[str]] | None = None,
) -> argparse.ArgumentParser:
    """A parser with one `--field-name` flag per field of the dataclass `config_cls`.

    Flags default to "not given" (`argument_default=SUPPRESS`), so `vars(parse_args())` holds
    only what the user passed. `bool` fields get `--flag` / `--no-flag`, `X | None` fields accept
    `none`. `choices` maps field names to their allowed values. The help text shows the field's
    default; the class is not instantiated, so required fields (invalid defaults) are fine.
    """
    parser = argparse.ArgumentParser(description=description, argument_default=argparse.SUPPRESS)
    hints = typing.get_type_hints(config_cls)
    for field in dataclasses.fields(config_cls):
        flag = "--" + field.name.replace("_", "-")
        default = field.default if field.default is not dataclasses.MISSING else None
        kind, optional = unwrap_optional(hints[field.name])
        help_text = f"(default: {default})"
        if kind is bool:
            parser.add_argument(
                flag, action=argparse.BooleanOptionalAction, dest=field.name, help=help_text
            )
        else:
            parse = optional_of(kind) if optional else kind
            kwargs = {}
            if choices and field.name in choices:
                kwargs["choices"] = choices[field.name]
            parser.add_argument(flag, type=parse, dest=field.name, help=help_text, **kwargs)
    return parser
