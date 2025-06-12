## ZML: Between JAX and llama.cpp

I recently had a chance to chat with someone in the industry about ZML, and that got me thinking more deeply about what the project is trying to do and how it fits into the ML ecosystem.

ZML is a machine learning inference framework written in Zig. The pitch is that you get a Python-free, deterministic, lower-level setup that's easier to reason about and maybe more portable across hardware targets. I was pretty intrigued by the idea at first. I've used Zig before and really liked it. It has this great combination of clarity, simplicity, and low-level control. I've also dabbled in ML tooling in other languages, and so when I saw a full framework trying to do ML in Zig, it caught my attention. It felt like the kind of thing I'd want to dig into. But as I started using it, I began to notice a few things that made me pause.

The first thing that stood out was that ZML doesn't use Zig's native build system. Instead, it relies on Bazel, likely to deal with dependencies like MLIR and OpenXLA, which come from massively complex C++ codebases. Still, it was a letdown. I had hoped for a simple, Zig-native build experience, but I ran into Bazel rules and external dependencies almost right away.

Then I started looking into what ZML has to offer. Since it wraps the same compiler backends that JAX uses, any potential performance gains weren't going to come from some new compiler or lower-level control. The only difference was to be made in the frontend: instead of writing models in Python, you write them in Zig. So to me, ZML started to feel like just a Zig frontend for JAX.

But many of the language design choices that make Zig great for many tasks (e.g., system programming), may not all be best fit for writing neural networks. You don't have operator overloading in Zig, for instance, so you can't write simple expressions like x + y, and instead you have to call methods instead.

I also began comparing ZML to other inference frameworks like llama.cpp, which although also written in a low-level language, keeps dependencies minimal and gives developers more direct control over execution. ZML, by contrast, abstracts much of the execution pipeline behind its compiler stack, making the hard parts inaccessible and limiting low-level control.

So, ZML at the moment seemed caught in a kind of awkward middle ground: lower level than PyTorch or JAX, but not low-level enough to enable fine-grained control over critical performance factors like kernel fusion or memory layout, unlike JAX’s Pallas. At the same time, you are giving up the flexibility and rich ecosystem that Python offers, without yet getting a clear payoff in return. There aren’t any public benchmarks, so it’s hard to tell if this tradeoff actually delivers better performance.

Finally, I questioned how hard it will be for ZML to keep up with the rapid pace of ML innovation. New transformer variants and optimization techniques are popping up constantly, with established frameworks like PyTorch and JAX quickly incorporating these advances, often because they serve as the initial development platforms for researchers. With ZML, I imagine even getting things like newer versions of flash attention to compile correctly through the current stack could take serious engineering work.

If I had to suggest something that might help, I'd say start with a single benchmark. Show how ZML handles attention or matmul compared to other frameworks. Ideally offer a simple Python wrapper so people can test it in a familiar environment without needing to learn Zig or Bazel. If ZML wants to gain traction, it needs to lower the barrier to trying it, and give people some clear, measurable reason to stick with it.

On that note, adding an einsum-style API for tensor operations could make a big difference. Porting models would feel more familiar, and you'd gain a more expressive way to define computation without giving up the benefits of Zig’s type system. It might also open the door to lightweight training or fine-tuning in the future, even if only for specific use cases like adapters or quantization-aware updates.

I'm still keeping an eye on the project, and I'd be happy to contribute if the right opportunity comes up. But right now, it feels like ZML is asking for a lot without yet showing that what you get in return is clearly better. Maybe that will change. I hope it does.

