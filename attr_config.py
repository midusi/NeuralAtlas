from captum.attr import Attribution
from captum._utils.typing import Module, TensorOrTupleOfTensorsGeneric, TargetType


class AttributionConfig:
    def __init__(
        self,
        attribution_class: Attribution,
        callback: callable = lambda x: x,
        **kwargs,
    ):
        self.attribution_class = attribution_class
        self.config = kwargs
        self.callback = callback

    def attribute(
        self,
        model: Module,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType,
    ) -> TensorOrTupleOfTensorsGeneric:
        layer = self.config.get("layer", None)
        if layer:
            return self.callback(
                self.attribution_class(model, layer=layer).attribute(
                    inputs=inputs,
                    target=target,
                )
            )
        else:
            return self.callback(
                self.attribution_class(model).attribute(
                    inputs=inputs,
                    target=target,
                    **self.config,
                )
            )

    def __str__(self):
        return f"{self.attribution_class.__name__}"

    def __repr__(self):
        return f"AttributionConfig(attribution_class={self.attribution_class}, config={self.config})"
