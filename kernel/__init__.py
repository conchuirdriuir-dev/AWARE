"""The relational kernel, copied from DARA (conchuirdriuir-dev/DARA, `src/dara/kernel/`, main at 95cea64, 2026-10-30)
for the toy world's stage 1 (PLAN.md, "Porting DARA, stage by stage"). Its equations are DARA's docs/KERNEL_v0_1.md:
structures, conjunctive patterns, one total code length, admission by a 2-bit margin. It reads no meaning: every input
is a structure of opaque symbols. The modules are DARA's, with only their imports renamed; DARA's rules about who may
hold beliefs do not come with them (DECISIONS.md, 2026-10-30). Not yet copied: latent (stage 2), frames (stage 3)."""
