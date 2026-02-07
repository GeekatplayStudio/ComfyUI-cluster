
import json

data_checkpoints = """
I. Photorealism & People
Photorealism | Flux.2-dev-fp16.safetensors | https://huggingface.co/black-forest-labs/FLUX.2-dev | hyper-realism, skin-texture, text-rendering, high-fidelity | Group A (Flux VAE / T5-CLIP)
Photorealism | RealVisXL_V6.0_B1.safetensors | https://civitai.com/models/150629 | raw-photo, dslr, 8k, professional-photography, anatomy | Group C (SDXL VAE / CLIP-L)
Photorealism | Juggernaut_XL_v10.safetensors | https://civitai.com/models/133005 | cinematic, lighting, generalist, commercial-photography | Group C (SDXL VAE)
Photorealism | Z-Image-Turbo-v1.safetensors | https://huggingface.co/Tongyi-MAI/Z-Image-Turbo | speed, realism, architectural, clean-lighting | Group B (Z-VAE / Z-CLIP)
Photorealism | epiCPhotoGasm_v2.safetensors | https://civitai.com/models/132632 | studio-lighting, portraits, soft-skin, bokeh | Group D (SD1.5 VAE)
Photorealism | Realistic_Vision_V7.0.safetensors | https://civitai.com/models/4201 | analog-film, retro, human-anatomy, natural-poses | Group D (SD1.5 VAE)
Photorealism | IC-Light-XL-Realistic.safetensors | https://huggingface.co/lllyasviel/ic-light | relighting, shadow-control, background-blending | Group C (SDXL VAE)
Photorealism | Sateluco-IOR-XL.safetensors | https://civitai.com/models/sateluco | extreme-clarity, high-dynamic-range, reflection-logic | Group C (SDXL VAE)
Photorealism | AnalogMadness_XL_v4.safetensors | https://civitai.com/models/analog-xl | film-grain, kodak-portra, fuji-film, nostalgia | Group C (SDXL VAE)
Photorealism | Photopedia_XL_v2.safetensors | https://civitai.com/models/photopedia | stock-photography, documentary-style, authentic | Group C (SDXL VAE)
Photorealism | Mobius_XL_v1.safetensors | https://civitai.com/models/mobius | lighting-consistency, character-depth, sharp-focus | Group C (SDXL VAE)
Photorealism | AbsoluteReality_v1.8.safetensors | https://civitai.com/models/81458 | sharp-details, natural-skin, non-ai-look | Group D (SD1.5 VAE)
Photorealism | ThinkDiffusionXL_v1.safetensors | https://civitai.com/models/thinkdiffusion | 4k-resolution, crisp-edges, studio-render | Group C (SDXL VAE)
Photorealism | NightVision_XL_v9.safetensors | https://civitai.com/models/nightvision | night-photography, low-light, street-lighting | Group C (SDXL VAE)
Photorealism | CyberRealistic_XL_v6.safetensors | https://civitai.com/models/cyberrealistic | hyper-detail, intricate-skin, photorealism | Group C (SDXL VAE)

II. Anime & Stylized
Anime | Illustrious-XL-v1.0.safetensors | https://civitai.com/models/illustrious | anime, high-res, cel-shaded, modern-anime | Group C (SDXL VAE)
Anime | PonyDiffusion_V7_XL.safetensors | https://civitai.com/models/257 | character-design, complex-poses, fandom, expressive | Group C (SDXL VAE)
Anime | AAM_XL_AnimeMix_v2.safetensors | https://civitai.com/models/anime-mix | vibrant, fantasy-anime, wallpaper-quality | Group C (SDXL VAE)
Anime | MeinaMix_V11.safetensors | https://civitai.com/models/meinamix | soft-anime, romantic-style, clean-lines | Group D (SD1.5 VAE)
Anime | Cetus-Mix-Final.safetensors | https://civitai.com/models/cetus | painterly-anime, 2.5D, semi-realistic | Group D (SD1.5 VAE)
Anime | NoobAI-XL-v1.safetensors | https://civitai.com/models/noobai | danbooru-tags, technical-prompts, precision | Group C (SDXL VAE)
Anime | Animagine_XL_v4.0.safetensors | https://civitai.com/models/animagine | consistent-character, high-quality-anime, official-art-style | Group C (SDXL VAE)
Anime | AbyssOrangeMix3_AOM3.safetensors | https://civitai.com/models/abyssorangemix | dark-anime, detailed-eyes, aesthetic | Group D (SD1.5 VAE)
Anime | Counterfeit-V3.0.safetensors | https://civitai.com/models/counterfeit | background-focused, atmospheric-anime | Group D (SD1.5 VAE)
Anime | BlueberryMix_XL.safetensors | https://civitai.com/models/blueberry | moe-style, soft-colors, kawaii | Group C (SDXL VAE)
Anime | Mistoon_Anime_v2.safetensors | https://civitai.com/models/mistoon | flat-color, vector-style, clean-anime | Group D (SD1.5 VAE)
Anime | Pastel-Mix-Stylized.safetensors | https://civitai.com/models/pastel-mix | pastel-colors, soft-lighting, artistic-anime | Group D (SD1.5 VAE)
Anime | Real-Anime-XL.safetensors | https://civitai.com/models/real-anime | semi-realistic, 3d-anime-hybrid, thick-shading | Group C (SDXL VAE)
Anime | HoloDayo_XL_v1.safetensors | https://civitai.com/models/holodayo | hololive-consistent, fanart-style, vibrant | Group C (SDXL VAE)
Anime | Waifu-Diffusion-XL.safetensors | https://huggingface.co/waifu-diffusion | base-anime, massive-dataset, legacy | Group C (SDXL VAE)

III. Artistic & Painting
Artistic | Painterly_XL_Check_v2.safetensors | https://civitai.com/models/painterly | oil-painting, canvas, impasto, brushstrokes | Group C (SDXL VAE)
Artistic | DreamShaper_XL_Turbo.safetensors | https://civitai.com/models/112902 | digital-art, fantasy, concept-art, versatile | Group C (SDXL VAE)
Artistic | Watercolor_Stylized_v3.safetensors | https://civitai.com/models/watercolor | water-wash, ink-bleed, traditional-media | Group C (SDXL VAE)
Artistic | Copax_TimeLess_XL_v9.safetensors | https://civitai.com/models/timeless | atmospheric, ethereal, fine-art | Group C (SDXL VAE)
Artistic | Oil_Brush_Pro_v2.safetensors | https://civitai.com/models/oilbrush | thick-paint, museum-quality, traditional | Group C (SDXL VAE)
Artistic | Arcane_Style_XL.safetensors | https://civitai.com/models/arcane | stylized, neon, paint-splatter, arcane-tv-style | Group C (SDXL VAE)
Artistic | Inkpunk_Diffusion_v2.safetensors | https://civitai.com/models/inkpunk | ink-sketch, messy-lines, cyberpunk-art | Group D (SD1.5 VAE)
Artistic | Abstract_Master_XL.safetensors | https://civitai.com/models/abstract | geometric, non-representative, fine-art-abstract | Group C (SDXL VAE)
Artistic | Charcoal_Pro_v1.safetensors | https://civitai.com/models/charcoal | black-and-white, hand-drawn, smudge-texture | Group C (SDXL VAE)
Artistic | Ukiyo-e_XL_v2.safetensors | https://civitai.com/models/ukiyoe | japanese-print, woodblock, historical-art | Group C (SDXL VAE)
Artistic | ComicMix_XL_v3.safetensors | https://civitai.com/models/comicmix | western-comic, marvel-style, clean-inking | Group C (SDXL VAE)
Artistic | Gouache_Style_XL.safetensors | https://civitai.com/models/gouache | matte-paint, thick-wash, opaque-watercolor | Group C (SDXL VAE)
Artistic | Pencil-Sketch-XL.safetensors | https://civitai.com/models/pencil-sketch | graphite, cross-hatching, realistic-sketch | Group C (SDXL VAE)
Artistic | Synthwave_Master_XL.safetensors | https://civitai.com/models/synthwave | 80s-aesthetic, retro-future, vibrant-pink-blue | Group C (SDXL VAE)
Artistic | Dark-Vibe-XL.safetensors | https://civitai.com/models/dark-vibe | gothic, macabre, detailed-darkness | Group C (SDXL VAE)

IV. Architecture & Interiors
Architecture | Modern_Architecture_XL.safetensors | https://civitai.com/models/arch | blueprints, glass, minimalism, luxury-homes | Group C (SDXL VAE)
Architecture | Interior_Design_Flux_v2.safetensors | https://civitai.com/models/interior | home-staging, photorealistic-rooms, furniture-detail | Group A (Flux VAE)
Architecture | Abandoned_Places_XL.safetensors | https://civitai.com/models/overgrown | ruins, nature-reclaimed, moss, urban-decay | Group C (SDXL VAE)
Architecture | CyberCity_XL_Pro.safetensors | https://civitai.com/models/cybercity | skyscrapers, neon, futuristic-urbanism | Group C (SDXL VAE)
Architecture | Brutalist_Logic_XL.safetensors | https://civitai.com/models/brutalist | concrete, geometric, stark-shadows | Group C (SDXL VAE)
Architecture | Gothic_Stone_Carving.safetensors | https://civitai.com/models/gothic | cathedral, intricate-stone, historical | Group C (SDXL VAE)
Architecture | TinyHouse_Master_XL.safetensors | https://civitai.com/models/tinyhouse | compact-living, modular, wood-texture | Group C (SDXL VAE)
Architecture | Steampunk_City_v2.safetensors | https://civitai.com/models/steamcity | brass, pipes, smoke, industrial-victorian | Group C (SDXL VAE)
Architecture | Skyscraper_High_Angle.safetensors | https://civitai.com/models/skyscraper | aerial-view, glass-reflections, urban-density | Group C (SDXL VAE)
Architecture | Japanese_Traditional_House.safetensors | https://civitai.com/models/japanesehouse | tatami, sliding-doors, zen-garden | Group C (SDXL VAE)

V. Landscapes & Environments
Landscape | Landscape_Anime_Max.safetensors | https://civitai.com/models/93931 | ghibli-style, lush-scenery, cloud-formations | Group C (SDXL VAE)
Landscape | GrandCanyon_RedRock_XL.safetensors | https://civitai.com/models/canyon | desert-textures, utah-scenery, sandstone | Group C (SDXL VAE)
Landscape | Misty_Forest_XL_v3.safetensors | https://civitai.com/models/misty | atmospheric-fog, pine-trees, moody-lighting | Group C (SDXL VAE)
Landscape | Arctic_Tundra_Pro.safetensors | https://civitai.com/models/arctic | snow-ice, blue-hour, aurora-borealis | Group C (SDXL VAE)
Landscape | Biolume_Jungle_XL.safetensors | https://civitai.com/models/biolume | glowing-plants, bioluminescence, exotic-nature | Group C (SDXL VAE)
Landscape | Volcano_Fury_XL.safetensors | https://civitai.com/models/volcano | lava, volcanic-smoke, dark-ash-sky | Group C (SDXL VAE)
Landscape | Alpine_Valley_v2.safetensors | https://civitai.com/models/alpine | mountains, green-valley, sharp-horizons | Group C (SDXL VAE)
Landscape | Lavender_Sunset_XL.safetensors | https://civitai.com/models/lavender | floral-fields, soft-sunset, horizon-line | Group C (SDXL VAE)
Landscape | Ocean_Deep_Pro.safetensors | https://civitai.com/models/ocean | underwater, coral-reefs, caustic-lighting | Group C (SDXL VAE)
Landscape | Martian_Colony_XL.safetensors | https://civitai.com/models/mars | red-dust, alien-sky, crater-landscape | Group C (SDXL VAE)

VI. Specialized & Niche (3D, Video, Utility)
Video | Wan_2.1_T2V_14B.safetensors | https://huggingface.co/Wan-AI/Wan2.1-T2V-14B | text-to-video, cinematic-motion, high-consistency | Group V (Wan VAE)
Video | HunyuanVideo_v1.safetensors | https://huggingface.co/tencent/HunyuanVideo | ai-video, long-duration, high-fps | Group V (Hunyuan VAE)
Video | SVD-XT-1.1.safetensors | https://huggingface.co/stabilityai/stable-video-diffusion | image-to-video, professional-motion | Group V (SVD VAE)
Video | LivePortrait_XL_v2.safetensors | https://civitai.com/models/liveportrait | facial-animation, vlog-automation, lip-sync | Group C (SDXL VAE)
3D Render | Disney_Pixar_Cartoon_XL.safetensors | https://civitai.com/models/75650 | 3d-animation, subsurface-scattering, toy-like | Group C (SDXL VAE)
3D Render | Blender-Style-XL.safetensors | https://civitai.com/models/blender | cycles-render, hard-surface, pbr-textures | Group C (SDXL VAE)
3D Render | Clay-Aardman-Style.safetensors | https://civitai.com/models/clay | stop-motion, fingerprint-texture, handmade | Group C (SDXL VAE)
3D Render | Unreal-Engine-5-XL.safetensors | https://civitai.com/models/ue5 | game-engine-lighting, high-poly, nanite-detail | Group C (SDXL VAE)
3D Render | ZBrush-Sculpt-XL.safetensors | https://civitai.com/models/zbrush | digital-sculpt, grey-clay, intricate-carving | Group C (SDXL VAE)
Utility | ControlNet-Union-SDXL.safetensors | https://huggingface.co/xinsir/controlnet-union | depth, canny, pose, all-in-one-control | Group C
Utility | IP-Adapter-FaceID-XL.safetensors | https://civitai.com/models/faceid | face-preservation, identity-consistency | Group C
Utility | OmniGen-v2.safetensors | https://huggingface.co/BectorSpaceLab/OmniGen-v2 | multi-image-composition, instruct-editing | Group A
Utility | Hyper-SDXL-Lightning.safetensors | https://huggingface.co/ByteDance/Hyper-SD | 1-step-generation, real-time-previews | Group C
Utility | Segment-Anything-2.safetensors | https://huggingface.co/facebook/sam2 | object-masking, surgical-selection, video-tracking | Group ALL
Photography | Food-Photography-XL.safetensors | https://civitai.com/models/food | restaurant-lighting, macro-food, professional-plating | Group C
Photography | Macro-Nature-XL.safetensors | https://civitai.com/models/macro | insect-detail, shallow-depth, dew-drops | Group C
Photography | Product-Commercial-XL.safetensors | https://civitai.com/models/product | clean-background, product-lighting, minimalist | Group C
Portrait | Mature-Faces-XL-v2.safetensors | https://civitai.com/models/silver | elderly, silver-hair, character-wrinkles | Group C
Portrait | Fashion-Editorial-XL.safetensors | https://civitai.com/models/fashion | vogue-style, high-contrast, model-poses | Group C
Niche | Tarot-Card-Check-XL.safetensors | https://civitai.com/models/tarot | mystical-border, symbolic-art, gold-leaf | Group C
Niche | Stained-Glass-v4.safetensors | https://civitai.com/models/glass | backlit, translucent, religious-art | Group C
Niche | Blueprint-Draft-XL.safetensors | https://civitai.com/models/blueprint | technical-drawing, grid-lines, industrial-design | Group C
Niche | Pixel-Art-Ultra-XL.safetensors | https://civitai.com/models/pixelart | retro-gaming, 64-bit, sprites | Group C
Niche | Graffiti-Wall-Pro-XL.safetensors | https://civitai.com/models/graffiti | spray-paint, street-art, tag-style | Group C
Niche | Chalkboard-Menu-XL.safetensors | https://civitai.com/models/chalk | dusty-chalk, handwritten, slate-texture | Group C
Niche | Embroidery-Craft-XL.safetensors | https://civitai.com/models/embroidery | sewn-texture, thread-patterns, fabric-art | Group C
Niche | Neon-Signs-XL.safetensors | https://civitai.com/models/neonsign | glowing-tubes, dark-city-streets | Group C
Niche | Infographic-Clean-XL.safetensors | https://civitai.com/models/info | charts, clean-typography, icons | Group C
Logic | Qwen2.5-VL-7B-Instruct.safetensors | https://huggingface.co/Qwen/Qwen2.5-VL | vlm, image-understanding, prompt-logic | Group L
Logic | Llama-4-8B-Vision.safetensors | https://huggingface.co/meta-llama/Llama-4 | reasoning, captioning, visual-logic | Group L
Niche | Medical-Anatomy-XL.safetensors | https://civitai.com/models/medical | scientific-illustration, biological-accuracy | Group C
Niche | Jewelry-Detail-XL.safetensors | https://civitai.com/models/jewelry | gold-reflections, gems, high-macro | Group C
Niche | Comic-Book-Inker-XL.safetensors | https://civitai.com/models/inker | thick-black-lines, cross-hatching | Group C
Niche | Retro-VCR-XL.safetensors | https://civitai.com/models/vcr | glitch-art, tracking-lines, 80s-video | Group C
Niche | Miniature-World-XL.safetensors | https://civitai.com/models/miniature | tilt-shift, toy-figures, macro-landscape | Group C
"""

data_loras = """
I. People & Realism
Realism | Boreal_Realism_Flux.safetensors | https://huggingface.co/kudzueye/Boreal | hyper-realism, candid, natural-skin | Group A
Realism | Smartphone_Snap_v2.safetensors | https://civitai.com/models/652699 | iphone-photo, grainy, vertical, casual | Group A
Realism | RealLife_Skin_Fix.safetensors | https://civitai.com/models/114321 | detailed-pores, realistic-lighting | Group C
Realism | Mature_Faces_Slider.safetensors | https://civitai.com/models/16543 | wrinkles, aging, silver-hair, character | Group C
Realism | Fashion_Editorial_Flux.safetensors | https://huggingface.co/alvdansen/flux-koda | vogue, high-fashion, studio-lighting | Group A
Realism | Cinematic_Shot_XL.safetensors | https://civitai.com/models/cinematic | anamorphic, color-grade, movie-look | Group C
Realism | Steve_McCurry_Photography.safetensors | https://civitai.com/models/mccurry | nat-geo, sharp-eyes, travel-photography | Group C
Realism | Low-Light_Night_Photography.safetensors | https://civitai.com/models/nightphoto | night-vision, low-noise, street-ambient | Group C
Realism | 8k_Ultra_Detail_Flux.safetensors | https://huggingface.co/prithivMLmods/Canopus | sharpness, intricate, masterpiece | Group A
Realism | Amateur_Style_Snapshot.safetensors | https://civitai.com/models/amateur | shaky-cam, low-quality-aesthetic, real-photo | Group C
Realism | Facial_Expression_Extreme.safetensors | https://civitai.com/models/extreme-face | angry, happy, crying, slider | Group A
Realism | Skin_Texture_Macro_Flux.safetensors | https://civitai.com/models/macro-skin | extreme-close-up, pores, hair-follicles | Group A
Realism | Candid_Street_Photography.safetensors | https://civitai.com/models/streetphoto | urban, crowd, natural-lighting | Group C
Realism | Kodak_Portra_400_XL.safetensors | https://civitai.com/models/kodak | warm-tones, film-grain, analog-photography | Group C
Realism | Hand_Fixer_v2.safetensors | https://civitai.com/models/handfix | perfect-fingers, anatomy-correction | Group C

II. Anime & Stylized Characters
Anime | Disney_Pixar_Style_v2.safetensors | https://civitai.com/models/459655 | disney, pixar, 3d-render, cartoon | Group C
Anime | Sakimi-Chan_Style_Flux.safetensors | https://civitai.com/models/21133 | digital-painting, vibrant, stylized-eyes | Group A
Anime | Makoto_Shinkai_Vibe.safetensors | https://civitai.com/models/shinkai | vibrant-sky, lens-flare, ghibli-hybrid | Group C
Anime | Arcane_Texture_XL.safetensors | https://civitai.com/models/arcane-lora | thick-paint, neon-accents, arcane-art | Group C
Anime | Spider-Verse_VFX.safetensors | https://civitai.com/models/spiderverse | chromatic-aberration, comic-dots | Group A
Anime | Retro_Anime_90s.safetensors | https://civitai.com/models/retro90s | vhs-aesthetic, scanlines, soft-colors | Group C
Anime | Claymation_Stop_Motion.safetensors | https://civitai.com/models/8876 | handmade, fingerprints, clay-texture | Group C
Anime | Papercraft_Origami_XL.safetensors | https://civitai.com/models/9987 | layered-paper, crafting, 3d-depth | Group C
Anime | Comic_Ink_Lineart.safetensors | https://civitai.com/models/lineart | manga-lineart, clean-inking, b&w | Group C
Anime | Ghibli_Background_Artist.safetensors | https://civitai.com/models/ghibliback | scenic, nature, watercolor-anime | Group C
Anime | Neon_Cyberpunk_Fashion.safetensors | https://civitai.com/models/neoncloth | glowing-fabric, techwear, futuristic | Group A
Anime | Elf_Ears_Fantasy.safetensors | https://civitai.com/models/elfears | pointed-ears, lotr-style, fantasy | Group C
Anime | Super_Hero_Suit_Texture.safetensors | https://civitai.com/models/supersuit | spandex, mask, metallic-armor | Group C
Anime | Cybernetic_Implants.safetensors | https://civitai.com/models/cybertech | augmented-parts, wires, metal-skin | Group A
Anime | Expressions_Anime_XL.safetensors | https://civitai.com/models/animeface | exaggerated, comedy, expressive | Group C

III. Artistic & Painting Styles
Painting | Oil_Impasto_Texture.safetensors | https://civitai.com/models/3030 | thick-paint, palette-knife, heavy-texture | Group C
Painting | Watercolor_Wash_Pro.safetensors | https://civitai.com/models/4040 | soft-edges, paper-texture, artistic-wash | Group C
Painting | Charcoal_Sketch_Flux.safetensors | https://huggingface.co/b_lora/charcoal | dark-lines, smudged, hand-drawn | Group A
Painting | Ukiyo-e_Traditional.safetensors | https://civitai.com/models/5050 | japanese-woodblock, flat-colors | Group C
Painting | Art_Nouveau_Mucha.safetensors | https://civitai.com/models/6060 | ornate, flowing-hair, organic-patterns | Group C
Painting | Synthwave_Grid_Vibe.safetensors | https://civitai.com/models/synthglow | 80s-retro, pink-grid, laser-beams | Group A
Painting | Gothic_Noir_XL.safetensors | https://civitai.com/models/noir | high-contrast, dark-mystery, moody | Group C
Painting | Minimalist_Vector_Art.safetensors | https://huggingface.co/renderartist/vector | clean-lines, flat-design, corporate | Group A
Painting | Steampunk_Gears_Brass.safetensors | https://civitai.com/models/9090 | victorian-tech, steam, pipes | Group C
Painting | Chalkboard_Art_Texture.safetensors | https://civitai.com/models/3236 | handwritten, dusty, white-chalk | Group A
Painting | Embroidery_Sewn_Art.safetensors | https://civitai.com/models/4347 | thread-work, stitch-patterns, fabric | Group C
Painting | Graffiti_Drip_Style.safetensors | https://civitai.com/models/2125 | urban-tags, paint-drips, street-art | Group C
Painting | Inkpunk_Splash_Art.safetensors | https://civitai.com/models/2020 | sketchy, industrial, ink-wash | Group C
Painting | Pastel_Dream_Vibe.safetensors | https://civitai.com/models/pastel | soft-colors, airy, ethereal | Group C
Painting | Pop_Art_Lichtenstein.safetensors | https://civitai.com/models/popart | halftone-dots, primary-colors, bold-inking | Group C

IV. Architecture & Environments
Architecture | Modern_Interior_Design.safetensors | https://civitai.com/models/1111 | minimalist, glass, luxury, staging | Group C
Architecture | Abandoned_Overgrown_Flux.safetensors | https://civitai.com/models/abandoned | ruins, nature-reclaimed, decay | Group A
Architecture | Cyber_City_Neon_Grid.safetensors | https://civitai.com/models/2222 | skyscrapers, neon-signs, futuristic | Group C
Architecture | Cozy_Cottage_Garden.safetensors | https://civitai.com/models/3333 | rural, warm-lighting, wood-textures | Group C
Architecture | Brutalist_Concrete_Flux.safetensors | https://civitai.com/models/4444 | stark-shadows, geometric, grey | Group A
Architecture | Gothic_Stone_Cathedral.safetensors | https://civitai.com/models/5555 | religious-art, intricate-carving | Group C
Architecture | Treehouse_Natural_Build.safetensors | https://civitai.com/models/6666 | hanging-structures, forest, wood | Group C
Architecture | Martian_Base_Modular.safetensors | https://civitai.com/models/7777 | scifi-domes, red-dust, futuristic | Group A
Architecture | Underwater_Biosphere.safetensors | https://civitai.com/models/8888 | glass-tunnels, coral, bioluminescence | Group C
Architecture | Tiny_House_Organization.safetensors | https://civitai.com/models/9999 | compact, smart-design, modern | Group C
Environment | Kitchen_Luxury_Flux.safetensors | https://civitai.com/models/4345 | marble-counters, high-end-appliances | Group A
Environment | Gamer_Room_RGB.safetensors | https://civitai.com/models/5456 | pc-setup, neon-strips, peripherals | Group C
Environment | Old_Library_Mahogany.safetensors | https://civitai.com/models/6567 | leather-books, dust-motes, wood-panels | Group C
Environment | Zen_Garden_Tranquil.safetensors | https://civitai.com/models/7678 | sand-patterns, bonsai, koi-pond | Group C
Environment | Industrial_Loft_Flux.safetensors | https://civitai.com/models/8789 | exposed-brick, high-ceilings, metal | Group A

V. Landscape & Photography Utility
Landscape | Arctic_Ice_Tundra.safetensors | https://civitai.com/models/1020 | snow-glare, blue-hour, cold | Group A
Landscape | Tropical_Island_Beach.safetensors | https://civitai.com/models/2030 | palm-trees, clear-water, sunny | Group C
Landscape | Grand_Canyon_Utah_XL.safetensors | https://civitai.com/models/3040 | red-rock, desert, high-dynamic-range | Group C
Landscape | Misty_Forest_Flux_Atmo.safetensors | https://civitai.com/models/4050 | fog, pine-trees, mysterious | Group A
Landscape | Volcano_Eruption_Fire.safetensors | https://civitai.com/models/5060 | lava, ash-cloud, cinematic-destruction | Group C
Landscape | Lavender_Field_Horizon.safetensors | https://civitai.com/models/6070 | purple, sunset-scenery, flora | Group C
Landscape | Space_Nebula_Starfield.safetensors | https://civitai.com/models/7080 | deep-space, galaxies, vibrant-colors | Group A
Landscape | Swiss_Alps_Summit.safetensors | https://civitai.com/models/8090 | mountain-peak, village, sharp-focus | Group C
Landscape | Autumn_Fall_Colors.safetensors | https://civitai.com/models/9001 | orange-leaves, crisp-air, seasonal | Group C
Landscape | Biolume_Crystal_Cave.safetensors | https://civitai.com/models/1002 | glow, fantasy-nature, subterranean | Group A
Utility | Detailifier_Enhance_XL.safetensors | https://civitai.com/models/126343 | sharpen, micro-detail, 8k-upscale | Group C
Utility | Lighting_Master_Pro.safetensors | https://civitai.com/models/2002 | rim-lighting, moody-shadows | Group C
Utility | No-Background_Sticker.safetensors | https://civitai.com/models/3003 | flat-background, cutout, white | Group A
Utility | Text_Spelling_Fixer.safetensors | https://civitai.com/models/4004 | coherent-text, correct-letters | Group A
Utility | Color_Pop_Vibrant.safetensors | https://civitai.com/models/1011 | saturation, vivid, contrast | Group A

VI. Niche, Clothes & Items (Remaining 25)
Clothing | Tactical_Armor_XL.safetensors | https://civitai.com/models/2022 | military-gear, straps, metal-texture | Group C
Clothing | Victorian_Lace_Dress.safetensors | https://civitai.com/models/3033 | historical-fabric, intricate-detail | Group C
Clothing | Silk_Satin_Flow_Flux.safetensors | https://civitai.com/models/2233 | reflective-fabric, soft-draping | Group A
Jewelry | Diamond_Gold_Macro.safetensors | https://civitai.com/models/5055 | reflective, gems, high-detail | Group C
Items | Sneaker_Head_Designer.safetensors | https://civitai.com/models/7077 | modern-shoes, street-fashion | Group A
Items | Mechanical_Clock_Gears.safetensors | https://civitai.com/models/gears | brass, intricate, movement | Group C
Items | Antique_Book_Binding.safetensors | https://civitai.com/models/oldbook | leather, gold-embossing, worn | Group C
Items | Glass_Orb_Reflection.safetensors | https://civitai.com/models/glassorb | distorted-view, caustic-light | Group C
Items | Potion_Bottle_Fantasy.safetensors | https://civitai.com/models/potion | liquid-fx, glowing, cork | Group C
VFX | Magic_Spell_Particles.safetensors | https://civitai.com/models/magic | glow, sparks, floating-runes | Group C
VFX | Glitch_Digital_Artifacts.safetensors | https://civitai.com/models/glitch | data-corruption, neon-smear | Group C
VFX | Smoke_Fog_Volumetric.safetensors | https://civitai.com/models/smokefx | swirling, density, realistic-mist | Group C
Niche | Tarot_Card_Frame.safetensors | https://civitai.com/models/5457 | mystical, border, card-art | Group A
Niche | Blueprint_Technical.safetensors | https://civitai.com/models/6568 | grid-lines, blue, drafting | Group C
Niche | Stained_Glass_Effect.safetensors | https://civitai.com/models/8790 | translucent, backlit, colorful | Group C
Niche | Origami_Paper_Fold.safetensors | https://civitai.com/models/9891 | geometric, sharp-creases | Group A
Niche | Tattoo_Traditional_Flash.safetensors | https://civitai.com/models/1014 | ink, white-background, classic | Group C
Niche | Neon_Sign_Glow.safetensors | https://civitai.com/models/5458 | night, glass-tubes, ambient-light | Group A
Niche | Miniature_Tilt_Shift.safetensors | https://civitai.com/models/tiltshift | diorama, blurred-top-bottom | Group C
Niche | Hologram_Projection.safetensors | https://civitai.com/models/holo | transparent, scan-lines, blue-glow | Group C
Niche | Infographic_Icon_Set.safetensors | https://civitai.com/models/icons | flat, minimalist, vector-symbols | Group C
Niche | Food_Plating_Gourmet.safetensors | https://civitai.com/models/plating | chef-style, fine-dining, sauce-drip | Group C
Niche | Sci-Fi_HUD_Overlay.safetensors | https://civitai.com/models/hud | user-interface, tech-graphics | Group C
Niche | Post-Apocalypse_Dust.safetensors | https://civitai.com/models/dust | hazy, gritty, sun-beams | Group C
Niche | Bioluminescent_Creatures.safetensors | https://civitai.com/models/biolumecreature | avatar-style, glowing-skin | Group C
"""

def parse_data(raw_text, item_type="checkpoint"):
    items = []
    lines = raw_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith("I"): continue # Skip headers/empty
        
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 5:
            category = parts[0]
            filename = parts[1]
            url = parts[2]
            tags = [t.strip() for t in parts[3].split(",")]
            group_info = parts[4].lower()
            
            # Simple type inference based on Group info
            model_type = "sdxl"
            if "flux" in group_info: model_type = "flux"
            elif "sd1.5" in group_info: model_type = "sd15"
            elif "svd" in group_info: model_type = "svd"
            elif "wan" in group_info: model_type = "wan"
            elif "hunyuan" in group_info: model_type = "hunyuan"
            
            item = {
                "name": filename,
                "category": category,
                "url": url,
                "tags": tags,
                "type": model_type,
                "group": parts[4], # Keep original group string
                "recommended": True
            }
            items.append(item)
    return items

checkpoints = parse_data(data_checkpoints, "checkpoint")
loras = parse_data(data_loras, "lora")

registry = {
    "version": 1,
    "checkpoints": checkpoints,
    "loras": loras
}

with open("d:\\ComfyUI\\0.10\\ComfyUI\\custom_nodes\\ComfyUI-Cluster\\model_registry.json", "w", encoding="utf-8") as f:
    json.dump(registry, f, indent=2)

print("Registry created successfully.")
