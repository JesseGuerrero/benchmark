# {Title}

**Abstract**: Land Surface Temperature (LST) is a key metric for heat island mitigation and cool urban planning meant to deter the effects of climate change for underrepresented areas. This work sets a free and open dataset benchmark for measuring the performance of models to predict a future LST by a monthly basis. According to an extensive literature review search, no other benchmarks exists for monthly temperature at a consistent moderate resolution of 30m^2. The dataset was scrapped from all U.S cities above 90 square miles resulting in DEM, Land Cover, NDBI, NDVI, NDWI, NDBI & LST for 107 cities. Metrics for temperature prediction include LST and a heat index 1-25 to generalize to individual cities. A baseline measurement was taken with the transformer architecture at 2.6 RMSE of a 1-25 Heat Index and 9.71F RMSE for LST prediction for all 107 cities in the United States. Surface temperature can be effectively predicted and generalized using only a few key variables. SOTA vision architecture, the choice of data and data augmentation contribute to effective pixel-wise prediction. 

<div align="center">
  <p float="left">
    <img src="https://i.imgur.com/AGNpQJa.png" width="32%" style="border: 1px solid #ddd; border-radius: 4px; margin: 0 0.5%; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);" />
    <img src="https://i.imgur.com/G735JWp.png" width="32%" style="border: 1px solid #ddd; border-radius: 4px; margin: 0 0.5%; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);" />
    <img src="https://i.imgur.com/dkjJvBC.png" width="32%" style="border: 1px solid #ddd; border-radius: 4px; margin: 0 0.5%; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);" />
  </p>
</div>

## Features

- Easily edit the content in Markdown instead of HTML.
- Quick-to-load, works with mobile devices, accessible, and SEO-friendly.
- Includes out-of-the-box components for the header, figures, image comparison sliders, LaTeX equations, two-column layouts, code blocks (with syntax highlighting), small caps, videos, and YouTube embeds.
- Add custom components using HTML or even other web frameworks like React, Vue, or Svelte.
- Built with [Astro](https://astro.build/), [Tailwind CSS](https://tailwindcss.com/), [MDX](https://mdxjs.com/), and [Typescript](https://www.typescriptlang.org/).

## Real-world Applications
- [Esri Maps SDK](https://jeremyiv.github.io/diffc-project-page/)
- [DeckGL](https://clip-rt.github.io/)
- [Cesium](https://stochsync.github.io/)
- [Omniverse](https://tbs-ualberta.github.io/CRESSim/)

## Usage

Want help setting it up? Please schedule a call with me [here](https://cal.com/romanhauksson/meeting), and I'll personally walk you through making your project page live! I want to talk to potential users to figure out pain points and features to add.

1. [Install Node.js](https://nodejs.org/en/download/package-manager).
1. Click "Use this template" to make a copy of this repository and then clone it, or just clone it directly.
1. Run `npm install` from the root of the project to install dependencies.
1. Edit the content in `/src/pages/index.mdx`, and remember to update the favicon and social link thumbnail (optional). In the frontmatter in `index.mdx`, they are set to `favicon.svg` and `screenshot-light.png` respectively, which refer to files in `/public/`.
1. Run `npm run dev` to see a live preview of your page while you edit it.
1. Host the website using [GitHub Pages](https://pages.github.com/), [Vercel](https://vercel.com), [Netlify](https://www.netlify.com/), or any other static site hosting service.

[![Deploy to Netlify](https://www.netlify.com/img/deploy/button.svg)](https://app.netlify.com/start/deploy?repository=https://github.com/romanhauksson/academic-project-astro-template) [![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FRomanHauksson%2Facademic-project-astro-template)

### Icons

This template uses [Astro Icon](https://www.astroicon.dev/) library.

To use a custom icon:

1. Search on [Iconify](https://icon-sets.iconify.design/) to find the icon you want. For example, the Hugging Face icon is `simple-icons:huggingface`, from the Simple Icons icon set.
1. Install the corresponding icon set: `npm install @iconify-json/simple-icons`.
1. If you're using the icon in one of the link buttons, add it in one of the objects in the `links` prop of the `<Header />` component in `index.mdx`:

```mdx
    {
      name: "Hugging Face",
      url: "https://huggingface.co/",
      icon: "simple-icons:huggingface"
    }
```

Or, to use it anywhere in an Astro component or MDX file:

```mdx
import { Icon } from "astro-icon/components";

<Icon name={"simple-icons:huggingface"} />
```

### Notes

- If you're using VS Code, I recommend installing the [Astro extension](https://marketplace.visualstudio.com/items?itemName=astro-build.astro-vscode) to get IntelliSense, syntax highlighting, and other features.
- When people share the link to your project on social media, it will often appear as a "link preview" based on the title, description, thumbnail, and favicon you configured. Double check that these previews look right using [this tool](https://linkpreview.xyz/).
- The Nerfies page uses the Google Sans font, which is licensed by Google, so unfortunately, I had to change it to a different font instead (I picked Noto Sans).

## Alternative templates

- [Clarity: A Minimalist Website Template for AI Research](https://shikun.io/projects/clarity) by Shikun Liu. Beautiful and careful design that's distinct from the original Nerfies page. Editable via an HTML template and SCSS.
- [Academic Project Page Template](https://denkiwakame.github.io/academic-project-template/) by Mai Nishimura. Built with React and UIKit and editable with Markdown in a YAML file.

## Credits

This template was adapted from Eliahu Horwitz's [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template), which was adapted from Keunhong Park's [project page for _Nerfies_](https://nerfies.github.io/). It's licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).
