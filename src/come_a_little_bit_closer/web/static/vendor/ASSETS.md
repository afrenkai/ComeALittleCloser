# Vendored browser assets

Downloaded on 2026-10-17. The explorer makes no request to an external tile server or JavaScript CDN.

| File | Source | License | SHA-256 |
| --- | --- | --- | --- |
| `htmx-2.0.4.min.js` | <https://raw.githubusercontent.com/bigskysoftware/htmx/v2.0.4/dist/htmx.min.js> | Zero-Clause BSD; see `HTMX-LICENSE` | `e209dda5c8235479f3166defc7750e1dbcd5a5c1808b7792fc2e6733768fb447` |

D3 7.9.0 is vendored from <https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js> for accurate orthographic projection and horizon clipping. License: ISC; see `D3-LICENSE`. SHA-256: `f2094bbf6141b359722c4fe454eb6c4b0f0e42cc10cc7af921fc158fceb86539`.

Country boundaries, downloaded 2026-10-17 from the official Natural Earth repository (public domain):

| File | Source | SHA-256 |
| --- | --- | --- |
| `ne_110m_admin_0_countries.geojson` | <https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson> | `6866c877d39cba9c357620878839b336d569f8c662d3cfab4cb1dbe2d39c977f` |
| `ne_50m_admin_0_countries.geojson` | <https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson> | `3e458fc036ad0a66411f2c1e6cac49c5d7bfb81cb1123bc513b22511a2b7fdeb` |

The globe starts with 1:110 million boundaries and switches to 1:50 million boundaries at 2.5× zoom. Country geometries are unmodified and projected on a sphere. The natural green/brown/ice palette is schematic, not a land-cover or elevation dataset. The two sources contain 177 and 242 map units respectively; differing resolution and map-unit coverage should not be interpreted as differing claims of sovereignty.
