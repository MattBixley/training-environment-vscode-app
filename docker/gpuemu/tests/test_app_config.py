"""The files Open OnDemand reads before a session ever starts.

None of this touches the emulator. It is here because a mistake in `form.yml`
does not fail loudly - the app simply does not appear on the dashboard, and the
next person to notice is whoever opens the training environment expecting a
workshop to be there.

The dashboard parses the form with `YAML.safe_load(contents)` and no
`aliases: true` (apps/dashboard/app/models/batch_connect/app.rb), so a YAML
anchor that is perfectly valid elsewhere raises Psych::AliasesNotEnabled and
takes the whole app down with it. That happened: v0.4.0 through v0.5.1 shipped
an anchored `options:` list and the app was missing for all three.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
FORM = REPO / "form.yml"
MANIFEST = REPO / "manifest.yml"
CARDS = ["l4", "a100_40", "a100", "h100", "pro_6000"]

# These files configure the app; they are not shipped inside the image, and the
# Dockerfile runs this suite from a copy of the package alone. Skipping there is
# correct - there is nothing to check - and the check-app-config CI job is what
# actually gates a release, so nothing is lost by the skip.
pytestmark = pytest.mark.skipif(
    not FORM.is_file(),
    reason=f"{FORM} not present: running from a copy of the package, not the repository",
)


class NoAliasLoader(yaml.SafeLoader):
    """Refuses aliases, the way Ruby's Psych does in safe_load."""

    def compose_node(self, parent, index):
        if self.check_event(yaml.events.AliasEvent):
            event = self.peek_event()
            raise AssertionError(
                f"{event.anchor!r} is a YAML alias. Open OnDemand parses this file "
                "with safe_load and no aliases, so the app will not load. Write the "
                "value out in full instead."
            )
        return super().compose_node(parent, index)


@pytest.mark.parametrize("path", [FORM, MANIFEST], ids=lambda p: p.name)
def test_open_ondemand_can_parse_it(path):
    assert path.is_file(), f"{path} is missing"
    loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=NoAliasLoader)
    assert isinstance(loaded, dict) and loaded


def _form() -> dict:
    return yaml.load(FORM.read_text(encoding="utf-8"), Loader=NoAliasLoader)


def test_every_control_on_the_form_is_defined():
    form = _form()
    missing = [k for k in form["form"] if k not in form["attributes"]]
    assert not missing, f"listed in form: but never defined: {missing}"


def test_there_is_one_memory_control_per_card():
    form = _form()
    assert [k for k in form["form"] if k.startswith("vram_")] == [f"vram_{c}" for c in CARDS]


@pytest.mark.parametrize("card", CARDS)
def test_each_card_offers_the_same_sizes_and_can_be_switched_off(card):
    """The options are repeated per card rather than shared, so they can drift."""
    attr = _form()["attributes"][f"vram_{card}"]
    values = [v for _label, v in attr["options"]]
    assert values == ["200MiB", "100MiB", "512MiB", "1GiB", "2GiB", "4GiB", "full", "off"]
    assert attr["value"] == "200MiB"
    assert attr["value"] in values, "the default must be one of the offered options"


def test_the_form_default_matches_the_emulator_default():
    """Two places name the card size; a session is confusing if they disagree."""
    from gpuemu.spec import DEFAULT_VRAM

    assert _form()["attributes"]["vram_l4"]["value"] == DEFAULT_VRAM


def test_the_emulator_switch_hides_every_card_while_off():
    """The card controls only appear once the emulators are switched On.

    Each Off-option key must be data-hide-<field with - for _>. A check box's
    data-hide-*-when-un-checked form cannot express vram_a100_40 or
    vram_pro_6000 (OnDemand's parser throws on the "-40"), so this must stay a
    select.
    """
    attr = _form()["attributes"]["gpu_emulator"]
    assert attr["widget"] == "select"
    options = {opt[1]: opt[2:] for opt in attr["options"]}
    assert set(options) == {"0", "1"} and attr["value"] == "0"
    off = {k: v for d in options["0"] for k, v in d.items()}
    expected = {f"data-hide-vram-{c.replace('_', '-')}": True for c in CARDS}
    assert off == expected
    assert options["1"] == [], "On must not hide anything"

