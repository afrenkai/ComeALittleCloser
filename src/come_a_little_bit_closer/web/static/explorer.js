(() => {
  const state = { view: "globe", rotation: null, globeZoom: 1, yaw: -.35, pitch: .2, scatterZoom: 1, scatterPanX: 0, scatterPanY: 0, drawerOpen: false, modelId: null };
  const requests = new Map();
  const countryColors = new WeakMap();
  const xhrSequences = new WeakMap();
  let cleanup = null;
  let playerSequence = 0;
  let requestSequence = 0;

  function playSelectedAudio() {
    const audio = document.getElementById("station-audio"), status = document.getElementById("player-status"), sequence = ++playerSequence;
    status.textContent = "Connecting…";
    document.getElementById("player-retry").hidden = true;
    audio.play().catch((error) => {
      if (sequence !== playerSequence) return;
      status.textContent = error.name === "NotAllowedError" ? "Press play" : "Stream unavailable";
      document.getElementById("player-retry").hidden = false;
    });
  }
  function selectStation(station) {
    const player = document.getElementById("station-player"), audio = document.getElementById("station-audio");
    document.getElementById("player-station-name").textContent = station.name;
    document.getElementById("player-station-country").textContent = station.country_code || "";
    player.hidden = false;
    let stream;
    try { stream = new URL(station.stream_url); } catch (error) { stream = null; }
    if (!stream || !["http:", "https:"].includes(stream.protocol)) {
      playerSequence += 1; audio.pause(); audio.removeAttribute("src"); audio.load();
      document.getElementById("player-status").textContent = "Stream unavailable";
      document.getElementById("player-retry").hidden = true;
      return;
    }
    if (audio.src !== stream.href) { audio.src = stream.href; audio.load(); }
    playSelectedAudio();
  }
  function initializePlayer() {
    const player = document.getElementById("station-player"), audio = document.getElementById("station-audio"), status = document.getElementById("player-status");
    audio.addEventListener("playing", () => { status.textContent = "Playing"; document.getElementById("player-retry").hidden = true; });
    audio.addEventListener("pause", () => { if (!player.hidden && !audio.error) status.textContent = "Paused"; });
    audio.addEventListener("waiting", () => { if (!player.hidden) status.textContent = "Buffering…"; });
    audio.addEventListener("error", () => { if (!player.hidden) { status.textContent = "Stream unavailable"; document.getElementById("player-retry").hidden = false; } });
    document.getElementById("player-close").addEventListener("click", () => { playerSequence += 1; player.hidden = true; audio.pause(); audio.removeAttribute("src"); audio.load(); });
    document.getElementById("player-retry").addEventListener("click", () => { audio.load(); playSelectedAudio(); });
  }
  function cachedJson(url) {
    if (!requests.has(url)) requests.set(url, fetch(url).then((response) => {
      if (!response.ok) throw new Error("Map data unavailable");
      return response.json();
    }).catch((error) => { requests.delete(url); throw error; }));
    return requests.get(url);
  }
  function updateGenreChip() {
    const genre = document.getElementById("genre").value.trim(), chip = document.getElementById("genre-chip");
    chip.hidden = genre.length === 0;
    document.getElementById("genre-chip-text").textContent = genre;
    chip.setAttribute("aria-label", `Clear ${genre} station tag filter`);
  }
  function formReady(form) {
    return Array.from(form.elements).every((control) => !control.willValidate || control.validity.valid);
  }
  function initializeControls() {
    const form = document.getElementById("search-form");
    form.addEventListener("invalid", (event) => {
      const advanced = event.target.closest(".advanced-controls");
      if (advanced) advanced.open = true;
    }, true);
    updateGenreChip();
    document.getElementById("genre").addEventListener("input", updateGenreChip);
    document.getElementById("genre-chip").addEventListener("click", () => {
      document.getElementById("genre").value = ""; updateGenreChip(); if (formReady(form)) form.requestSubmit();
    });
    document.getElementById("model").addEventListener("change", () => { if (formReady(form)) form.requestSubmit(); });
  }

  function initializeComparison(container) {
    if (!container || container.dataset.initialized) return;
    if (cleanup) cleanup();
    container.dataset.initialized = "true";
    const searchResult = JSON.parse(container.querySelector("[data-map-data]").textContent);
    const result = searchResult || { origin: null, geographic_neighbors: [], embedding_neighbors: [], projection: null, model_id: null };
    const canvas = container.querySelector("[data-world-map]"), context = canvas.getContext("2d"), stage = container.querySelector("[data-map-stage]");
    const tooltip = container.querySelector("[data-map-tooltip]"), loading = container.querySelector("[data-map-loading]"), empty = container.querySelector("[data-projection-empty]");
    const note = container.querySelector("[data-projection-note]"), drawer = container.querySelector("[data-results-drawer]"), rows = Array.from(container.querySelectorAll("[data-station-row]"));
    const projectionData = result.projection || null;
    const projectionPoints = projectionData ? projectionData.points : [];
    const geographicIds = new Set(result.geographic_neighbors.map((station) => station.station_id)), embeddingIds = new Set(result.embedding_neighbors.map((station) => station.station_id));
    const stations = new Map();
    projectionPoints.forEach((point) => stations.set(point.station_id, point));
    result.geographic_neighbors.concat(result.embedding_neighbors, result.unfiltered_embedding_neighbors || []).forEach((station) => stations.set(station.station_id, { ...stations.get(station.station_id), ...station }));
    let globeStations = new Map(stations);
    let geographicOrigin = result.origin ? [result.origin.longitude, result.origin.latitude] : null;
    if (!state.rotation) state.rotation = [0, -10, 0];
    if (state.modelId !== result.model_id) { state.modelId = result.model_id; state.yaw = -.35; state.pitch = .2; state.scatterZoom = 1; state.scatterPanX = 0; state.scatterPanY = 0; }
    const frameCenter = projectionData ? projectionData.frame_center : { x: 0, y: 0, z: 0 };
    const frameRadius = Math.max(projectionData ? projectionData.frame_radius : 1, 1e-9);
    const clusterColors = d3.schemeTableau10;
    const ringColors = { geographic: "#77caff", embedding: "#ffae78", shared: "#ecf2b2", other: "#d7e9cd" };
    const pointers = new Map();
    let width = 0, height = 0, countries = null, fineCountries = null, loadingFine = false, activeStation = null, hitPoints = [], drag = null, pinch = null, moved = false, drawFrame = null;
    const globe = d3.geoOrthographic().clipAngle(90).precision(.25), path = d3.geoPath(globe, context), graticule = d3.geoGraticule10();
    if (drawer) { drawer.open = state.drawerOpen; drawer.addEventListener("toggle", () => { state.drawerOpen = drawer.open; }); }

    function category(stationId) {
      if (geographicIds.has(stationId) && embeddingIds.has(stationId)) return "shared";
      if (geographicIds.has(stationId)) return "geographic";
      if (embeddingIds.has(stationId)) return "embedding";
      return "other";
    }
    function globeVisible(coordinates) { return d3.geoDistance([-state.rotation[0], -state.rotation[1]], coordinates) < Math.PI / 2 - .00001; }
    function plotFrame() { return { left: 35, right: width - 35, top: 100, bottom: Math.max(190, height - (width < 800 ? 210 : 160)) }; }
    function scatterCenter() { const frame = plotFrame(); return [width / 2 + state.scatterPanX * width, (frame.top + frame.bottom) / 2 + state.scatterPanY * height]; }
    function rotateVector(vector) {
      const yawCos = Math.cos(state.yaw), yawSin = Math.sin(state.yaw), pitchCos = Math.cos(state.pitch), pitchSin = Math.sin(state.pitch);
      const x = yawCos * vector[0] + yawSin * vector[2], z = -yawSin * vector[0] + yawCos * vector[2];
      return [x, pitchCos * vector[1] - pitchSin * z, pitchSin * vector[1] + pitchCos * z];
    }
    function scatterPosition(point) {
      const rotated = rotateVector([(point.x - frameCenter.x) / frameRadius, (point.y - frameCenter.y) / frameRadius, (point.z - frameCenter.z) / frameRadius]);
      const depth = 3.4 - rotated[2];
      if (depth <= .15) return null;
      const perspective = 3.4 / depth, frame = plotFrame(), center = scatterCenter(), scale = Math.min(frame.right - frame.left, frame.bottom - frame.top) * .44 * state.scatterZoom;
      return { x: center[0] + rotated[0] * scale * perspective, y: center[1] - rotated[1] * scale * perspective, depth: rotated[2], perspective };
    }
    function marker(x, y, color, radius, ring = null, active = false) {
      context.beginPath(); context.arc(x, y, radius, 0, Math.PI * 2); context.fillStyle = color; context.fill();
      if (ring || active) { context.beginPath(); context.arc(x, y, radius + 2.3, 0, Math.PI * 2); context.strokeStyle = active ? "#ffffff" : ring; context.lineWidth = active ? 1.8 : 1.5; context.stroke(); }
    }
    function originMarker(position) {
      if (!position) return;
      const x = position.x === undefined ? position[0] : position.x, y = position.y === undefined ? position[1] : position.y;
      context.beginPath(); context.arc(x, y, 12, 0, Math.PI * 2); context.strokeStyle = "#f4f6e870"; context.lineWidth = 1; context.stroke();
      context.beginPath(); context.moveTo(x, y - 5); context.lineTo(x + 5, y); context.lineTo(x, y + 5); context.lineTo(x - 5, y); context.closePath(); context.fillStyle = "#ffffef"; context.fill();
    }
    function countryColor(feature) {
      if (countryColors.has(feature)) return countryColors.get(feature);
      const code = feature.properties.ADM0_A3;
      const center = d3.geoCentroid(feature), latitude = center[1], longitude = center[0];
      let color = "#829762";
      if (code === "ATA" || code === "GRL") color = "#e4e8dd";
      else if (latitude > 15 && latitude < 35 && longitude > -20 && longitude < 65) color = "#b7a776";
      else if (latitude < -18 && latitude > -34 && longitude > 112) color = "#b09b6c";
      else if (Math.abs(latitude) > 55) color = "#90a17a";
      else if (Math.abs(latitude) < 15) color = "#648d55";
      countryColors.set(feature, color);
      return color;
    }
    function requestFineCountries() {
      if (fineCountries || loadingFine || state.globeZoom < 2.5) return;
      loadingFine = true;
      cachedJson("/static/vendor/ne_50m_admin_0_countries.geojson").then((collection) => { if (!canvas.isConnected) return; fineCountries = collection; requestDraw(); }).catch(() => { loadingFine = false; });
    }
    function drawGlobe() {
      globe.translate([width / 2, height * .47]).scale(Math.min(width * .37, height * .385) * state.globeZoom).rotate(state.rotation);
      const sphere = { type: "Sphere" };
      context.save();
      context.shadowColor = "#4c8fbd35"; context.shadowBlur = 40;
      context.beginPath(); path(sphere);
      const ocean = context.createLinearGradient(0, height * .15, width * .8, height * .8); ocean.addColorStop(0, "#377baf"); ocean.addColorStop(1, "#174e79"); context.fillStyle = ocean; context.fill(); context.shadowBlur = 0;
      context.beginPath(); path(graticule); context.strokeStyle = "#c1d9e126"; context.lineWidth = .6; context.stroke();
      const visibleCountries = state.globeZoom >= 2.5 && fineCountries ? fineCountries : countries;
      if (visibleCountries) visibleCountries.features.forEach((feature) => {
        context.beginPath(); path(feature); context.fillStyle = countryColor(feature); context.fill(); context.strokeStyle = "#3b553c9e"; context.lineWidth = state.globeZoom > 8 ? .9 : .55; context.stroke();
      });
      context.beginPath(); path(sphere); context.strokeStyle = "#b3d0d47d"; context.lineWidth = 1; context.stroke();
      context.restore();
      requestFineCountries();
      stations.forEach((station, stationId) => {
        if (!geographicOrigin || category(stationId) === "other") return;
        context.beginPath(); path({ type: "LineString", coordinates: [geographicOrigin, [station.longitude, station.latitude]] });
        context.strokeStyle = stationId === activeStation ? "#f5f3d195" : "#eff4d32c"; context.lineWidth = stationId === activeStation ? 1.3 : .8; context.stroke();
      });
      const foreground = [];
      globeStations.forEach((station, stationId) => {
        const coordinates = [station.longitude, station.latitude];
        if (!globeVisible(coordinates)) return;
        const position = globe(coordinates);
        if (position[0] < -15 || position[0] > width + 15 || position[1] < -15 || position[1] > height + 15) return;
        const group = category(stationId), active = stationId === activeStation;
        if (group === "other" && !active) marker(position[0], position[1], "#e4efd5b8", state.globeZoom > 8 ? 2 : 1.5);
        else foreground.push({ stationId, position, group, active });
        hitPoints.push({ station, x: position[0], y: position[1], selected: group !== "other", depth: 0 });
      });
      foreground.forEach((point) => marker(point.position[0], point.position[1], ringColors[point.group], point.active ? 5 : 3.4, "#12302a", point.active));
      if (geographicOrigin && globeVisible(geographicOrigin)) originMarker(globe(geographicOrigin));
    }
    function drawTriad() {
      const frame = plotFrame(), center = [frame.left + 32, frame.bottom - 20];
      const axes = [{ vector: [1, 0, 0], label: "PC 1", color: "#e58f88" }, { vector: [0, 1, 0], label: "PC 2", color: "#9bc78a" }, { vector: [0, 0, 1], label: "PC 3", color: "#82afd4" }];
      context.font = "9px Arial, sans-serif";
      axes.forEach((axis) => { const rotated = rotateVector(axis.vector), x = center[0] + rotated[0] * 29, y = center[1] - rotated[1] * 29; context.beginPath(); context.moveTo(center[0], center[1]); context.lineTo(x, y); context.strokeStyle = axis.color; context.lineWidth = 1.4; context.stroke(); context.fillStyle = axis.color; context.textAlign = rotated[0] < 0 ? "right" : "left"; context.fillText(axis.label, x + (rotated[0] < 0 ? -5 : 5), y + 3); });
    }
    function drawScatter() {
      if (!projectionData) return;
      const projected = projectionPoints.map((station) => ({ station, position: scatterPosition(station) })).filter((point) => point.position && point.position.x > -15 && point.position.x < width + 15 && point.position.y > -15 && point.position.y < height + 15);
      projected.sort((left, right) => left.position.depth - right.position.depth);
      projected.forEach((point) => {
        const station = point.station, position = point.position, group = category(station.station_id), active = station.station_id === activeStation;
        const color = clusterColors[((station.cluster_id % clusterColors.length) + clusterColors.length) % clusterColors.length];
        context.globalAlpha = active || group !== "other" ? 1 : Math.max(.4, Math.min(.9, .67 + position.depth * .18));
        const radius = (active ? 5 : group === "other" ? 2 : 3.2) * Math.max(.75, Math.min(1.5, position.perspective));
        marker(position.x, position.y, color, radius, group === "other" ? null : ringColors[group], active);
        context.globalAlpha = 1;
        hitPoints.push({ station, x: position.x, y: position.y, selected: group !== "other", depth: position.depth });
      });
      originMarker(scatterPosition(projectionData.query));
      drawTriad();
    }
    function draw() {
      drawFrame = null;
      if (!canvas.isConnected || !width || !height) return;
      context.clearRect(0, 0, width, height); hitPoints = [];
      if (state.view === "globe") drawGlobe(); else drawScatter();
    }
    function requestDraw() { if (drawFrame === null) drawFrame = requestAnimationFrame(draw); }
    function resize() {
      const bounds = stage.getBoundingClientRect(); width = bounds.width; height = bounds.height;
      const resolution = Math.min(window.devicePixelRatio || 1, 2); canvas.width = Math.round(width * resolution); canvas.height = Math.round(height * resolution); context.setTransform(resolution, 0, 0, resolution, 0, 0); requestDraw();
    }
    function setActive(stationId) { if (activeStation === stationId) return; activeStation = stationId; rows.forEach((row) => row.classList.toggle("is-active", row.dataset.stationRow === stationId)); requestDraw(); }
    function tooltipFor(station, x, y) {
      const group = category(station.station_id), label = group === "shared" ? "Shared neighbor" : group === "geographic" ? "Geographic neighbor" : group === "embedding" ? "Embedding neighbor" : "Station";
      const distance = Number.isFinite(station.distance_km) ? `${station.distance_km.toLocaleString(undefined, { maximumFractionDigits: 1 })} km` : "", similarity = Number.isFinite(station.similarity) ? `cosine ${station.similarity.toFixed(4)}` : "";
      const metrics = [distance, similarity].filter((metric) => metric.length > 0).join(" · ");
      tooltip.textContent = `${station.name}\n${station.country_code || "—"} · ${label}${metrics ? `\n${metrics}` : ""}`;
      tooltip.hidden = false; tooltip.style.left = `${Math.max(10, Math.min(width - tooltip.offsetWidth - 10, x + 15))}px`; tooltip.style.top = `${Math.max(10, Math.min(height - tooltip.offsetHeight - 10, y + 15))}px`;
    }
    function nearest(x, y) {
      let closest = null, distance = 14;
      hitPoints.forEach((point) => { const candidate = Math.hypot(point.x - x, point.y - y); if (candidate < distance || (point.selected && candidate < 8 && closest && !closest.selected)) { distance = candidate; closest = point; } });
      return closest;
    }
    function setView(view) {
      state.view = view;
      container.querySelectorAll("[data-view]").forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.view === view)));
      const embedding = view === "embedding";
      container.querySelector("[data-view-label]").textContent = embedding ? "3D embedding space" : "Geographic space";
      container.querySelector("[data-view-help]").textContent = embedding ? "Drag to orbit; Shift-drag or right-drag to pan. Scroll or pinch to zoom." : "Orthographic globe with Natural Earth country boundaries. Schematic natural land palette; country shapes are unmodified.";
      empty.hidden = !embedding || Boolean(projectionData); loading.hidden = embedding || Boolean(countries); note.hidden = !embedding || !projectionData;
      const legend = container.querySelector("[data-cluster-legend]"); legend.replaceChildren(); legend.hidden = !embedding || !projectionData;
      if (embedding && projectionData) {
        const variance = projectionData.explained_variance_ratio.reduce((sum, value) => sum + value, 0) * 100;
        note.textContent = `${projectionData.description} · ${variance.toFixed(1)}% variance in 3D. Projection can distort distances; retrieval uses full embeddings. Colors show full-dimensional k-means groups, with metadata-derived labels.`;
        projectionData.clusters.forEach((cluster) => { const item = document.createElement("span"), swatch = document.createElement("i"), label = document.createElement("span"); swatch.style.backgroundColor = clusterColors[((cluster.cluster_id % clusterColors.length) + clusterColors.length) % clusterColors.length]; label.textContent = cluster.label; item.append(swatch, label); legend.append(item); });
      }
      canvas.setAttribute("aria-label", embedding ? "Interactive three-dimensional PCA projection of actual embeddings. Drag or arrow keys to orbit, Shift-drag or right-drag to pan, scroll or pinch to zoom." : "Interactive orthographic globe. Drag to rotate, click a station to listen, click the visible globe to set origin, scroll or pinch to zoom.");
      tooltip.hidden = true; requestDraw();
    }
    function keepGlobeAnchor(coordinates, anchor) {
      for (let iteration = 0; iteration < 5; iteration += 1) {
        globe.rotate(state.rotation);
        const point = globe(coordinates), errorX = anchor.x - point[0], errorY = anchor.y - point[1];
        if (Math.hypot(errorX, errorY) < .1) break;
        const step = .001;
        globe.rotate([state.rotation[0] + step, state.rotation[1], 0]); const dx = globe(coordinates);
        globe.rotate([state.rotation[0], state.rotation[1] + step, 0]); const dy = globe(coordinates);
        const a = (dx[0] - point[0]) / step, b = (dy[0] - point[0]) / step, c = (dx[1] - point[1]) / step, d = (dy[1] - point[1]) / step, determinant = a * d - b * c;
        if (Math.abs(determinant) < 1e-8) break;
        state.rotation[0] += Math.max(-20, Math.min(20, (errorX * d - b * errorY) / determinant));
        state.rotation[1] = Math.max(-89.99, Math.min(89.99, state.rotation[1] + Math.max(-20, Math.min(20, (a * errorY - errorX * c) / determinant))));
      }
      globe.rotate(state.rotation);
    }
    function zoom(factor, anchor = null) {
      if (state.view === "globe") {
        const center = globe.translate(), oldRadius = globe.scale();
        const inside = anchor && Math.hypot(anchor.x - center[0], anchor.y - center[1]) < oldRadius * .98;
        const coordinates = inside ? globe.invert([anchor.x, anchor.y]) : null;
        const oldZoom = state.globeZoom; state.globeZoom = Math.max(.65, Math.min(128, oldZoom * factor)); globe.scale(oldRadius * state.globeZoom / oldZoom);
        if (coordinates && coordinates.every(Number.isFinite) && Math.hypot(anchor.x - center[0], anchor.y - center[1]) < globe.scale() * .98) keepGlobeAnchor(coordinates, anchor);
      } else {
        const oldZoom = state.scatterZoom, center = scatterCenter(); state.scatterZoom = Math.max(.25, Math.min(128, oldZoom * factor));
        if (anchor) { const ratio = state.scatterZoom / oldZoom; state.scatterPanX += (anchor.x - center[0]) * (1 - ratio) / width; state.scatterPanY += (anchor.y - center[1]) * (1 - ratio) / height; }
      }
      tooltip.hidden = true; requestDraw();
    }
    function pointerLocation(event) { const bounds = canvas.getBoundingClientRect(); return { x: event.clientX - bounds.left, y: event.clientY - bounds.top }; }
    function beginDrag(point, pan) { return { x: point.x, y: point.y, rotation: state.rotation.slice(), yaw: state.yaw, pitch: state.pitch, panX: state.scatterPanX, panY: state.scatterPanY, pan }; }
    function pinchState() { const points = Array.from(pointers.values()); return { distance: Math.hypot(points[0].x - points[1].x, points[0].y - points[1].y), x: (points[0].x + points[1].x) / 2, y: (points[0].y + points[1].y) / 2 }; }
    container.querySelectorAll("[data-view]").forEach((button) => button.addEventListener("click", () => setView(button.dataset.view)));
    container.querySelectorAll("[data-map-zoom]").forEach((button) => button.addEventListener("click", () => zoom(button.dataset.mapZoom === "in" ? 1.5 : 1 / 1.5)));
    container.querySelector("[data-map-reset]").addEventListener("click", () => { if (state.view === "globe") { state.rotation = geographicOrigin ? [-geographicOrigin[0], -geographicOrigin[1], 0] : [0, -10, 0]; state.globeZoom = 1; } else { state.yaw = -.35; state.pitch = .2; state.scatterZoom = 1; state.scatterPanX = 0; state.scatterPanY = 0; } tooltip.hidden = true; requestDraw(); });
    container.querySelectorAll("[data-focus-station]").forEach((button) => {
      const station = stations.get(button.dataset.focusStation);
      button.addEventListener("mouseenter", () => setActive(station.station_id)); button.addEventListener("mouseleave", () => setActive(null)); button.addEventListener("focus", () => setActive(station.station_id)); button.addEventListener("blur", () => setActive(null));
      button.addEventListener("click", () => { if (state.view === "globe") state.rotation = [-station.longitude, -station.latitude, 0]; else if (typeof station.z === "number") { const position = scatterPosition(station), center = scatterCenter(); if (position) { state.scatterPanX -= (position.x - center[0]) / width; state.scatterPanY -= (position.y - center[1]) / height; } } setActive(station.station_id); requestDraw(); selectStation(station); });
    });
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    canvas.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 && event.button !== 2) return;
      event.preventDefault(); const point = pointerLocation(event); pointers.set(event.pointerId, point); canvas.setPointerCapture(event.pointerId);
      if (pointers.size === 1) { drag = beginDrag(point, event.shiftKey || event.button === 2); moved = false; }
      else if (pointers.size === 2) { pinch = pinchState(); drag = null; moved = true; }
    });
    canvas.addEventListener("pointermove", (event) => {
      const point = pointerLocation(event);
      if (pointers.has(event.pointerId)) pointers.set(event.pointerId, point);
      if (pointers.size >= 2 && pinch) {
        const current = pinchState(); moved = true; tooltip.hidden = true;
        zoom(current.distance / Math.max(1, pinch.distance), current);
        if (state.view === "embedding") { state.scatterPanX += (current.x - pinch.x) / width; state.scatterPanY += (current.y - pinch.y) / height; }
        else { const sensitivity = 180 / (Math.PI * globe.scale()); state.rotation[0] += (current.x - pinch.x) * sensitivity; state.rotation[1] = Math.max(-89.99, Math.min(89.99, state.rotation[1] - (current.y - pinch.y) * sensitivity)); }
        pinch = current; requestDraw(); return;
      }
      if (drag) {
        const dx = point.x - drag.x, dy = point.y - drag.y;
        if (Math.hypot(dx, dy) < 3 && !moved) return;
        moved = true; canvas.classList.add("is-dragging"); tooltip.hidden = true;
        if (state.view === "globe") { const sensitivity = 180 / (Math.PI * globe.scale()); state.rotation = [drag.rotation[0] + dx * sensitivity, Math.max(-89.99, Math.min(89.99, drag.rotation[1] - dy * sensitivity)), 0]; }
        else if (drag.pan || event.shiftKey) { state.scatterPanX = drag.panX + dx / width; state.scatterPanY = drag.panY + dy / height; }
        else { state.yaw = drag.yaw + dx * .007; state.pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, drag.pitch + dy * .007)); }
        requestDraw(); return;
      }
      const nearestPoint = nearest(point.x, point.y);
      if (nearestPoint) { setActive(nearestPoint.station.station_id); tooltipFor(nearestPoint.station, point.x, point.y); canvas.style.cursor = "pointer"; }
      else { setActive(null); tooltip.hidden = true; canvas.style.cursor = "grab"; }
    });
    function stopPointer(event) {
      pointers.delete(event.pointerId); pinch = null;
      if (pointers.size === 1) drag = beginDrag(Array.from(pointers.values())[0], false); else drag = null;
      canvas.classList.remove("is-dragging");
    }
    canvas.addEventListener("pointerup", stopPointer); canvas.addEventListener("pointercancel", stopPointer);
    canvas.addEventListener("pointerleave", () => { if (!drag && !pinch) { setActive(null); tooltip.hidden = true; } });
    canvas.addEventListener("click", (event) => {
      if (moved) { moved = false; return; }
      const point = pointerLocation(event), station = nearest(point.x, point.y);
      if (station) { setActive(station.station.station_id); tooltipFor(station.station, point.x, point.y); selectStation(station.station); return; }
      if (state.view !== "globe") return;
      const center = globe.translate(); if (Math.hypot(point.x - center[0], point.y - center[1]) > globe.scale()) return;
      const coordinates = globe.invert([point.x, point.y]); if (!coordinates || !coordinates.every(Number.isFinite)) return;
      document.getElementById("latitude").value = coordinates[1].toFixed(5); document.getElementById("longitude").value = coordinates[0].toFixed(5); geographicOrigin = coordinates; requestDraw();
      const form = document.getElementById("search-form"); if (formReady(form)) form.requestSubmit();
    });
    canvas.addEventListener("wheel", (event) => { event.preventDefault(); const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? height : 1; const factor = Math.exp(-event.deltaY * unit * (event.ctrlKey ? .008 : .0015)); zoom(Math.max(.5, Math.min(2, factor)), pointerLocation(event)); }, { passive: false });
    canvas.addEventListener("keydown", (event) => {
      if (["+", "="].includes(event.key)) zoom(1.5); else if (event.key === "-") zoom(1 / 1.5);
      else if (event.key.startsWith("Arrow")) {
        const horizontal = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0, vertical = event.key === "ArrowDown" ? 1 : event.key === "ArrowUp" ? -1 : 0;
        if (state.view === "globe") { state.rotation[0] += horizontal * 8 / state.globeZoom; state.rotation[1] = Math.max(-89.99, Math.min(89.99, state.rotation[1] - vertical * 8 / state.globeZoom)); }
        else if (event.shiftKey) { state.scatterPanX += horizontal * .05; state.scatterPanY += vertical * .05; }
        else { state.yaw += horizontal * .1; state.pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, state.pitch + vertical * .1)); }
        requestDraw();
      } else return;
      event.preventDefault();
    });
    const observer = new ResizeObserver(resize); observer.observe(stage);
    cleanup = () => { observer.disconnect(); if (drawFrame !== null) cancelAnimationFrame(drawFrame); };
    resize(); setView(state.view);
    cachedJson("/static/vendor/ne_110m_admin_0_countries.geojson").then((collection) => { if (!canvas.isConnected) return; countries = collection; loading.hidden = true; requestDraw(); }).catch(() => { loading.textContent = "Country outlines unavailable"; });
    cachedJson("/api/stations").then((catalog) => { if (!canvas.isConnected || !Array.isArray(catalog)) return; globeStations = new Map(catalog.map((station) => [station.station_id, station])); stations.forEach((station, stationId) => globeStations.set(stationId, { ...globeStations.get(stationId), ...station })); requestDraw(); }).catch(() => { if (canvas.isConnected) { loading.textContent = "Station catalog unavailable"; loading.hidden = state.view !== "globe"; } });
  }
  document.addEventListener("DOMContentLoaded", () => { initializePlayer(); initializeControls(); initializeComparison(document.querySelector("[data-comparison]")); });
  document.addEventListener("htmx:beforeRequest", (event) => {
    if (event.detail.elt.id !== "search-form") return;
    requestSequence += 1; xhrSequences.set(event.detail.xhr, requestSequence); state.drawerOpen = true;
    document.getElementById("request-error").hidden = true; document.getElementById("results").setAttribute("aria-busy", "true"); updateGenreChip();
  });
  document.addEventListener("htmx:beforeSwap", (event) => { const sequence = xhrSequences.get(event.detail.xhr); if (event.detail.target.id === "results" && sequence !== undefined && sequence !== requestSequence) { event.detail.shouldSwap = false; event.preventDefault(); } });
  document.addEventListener("htmx:afterSwap", (event) => { if (event.detail.target.id === "results") initializeComparison(event.detail.target.querySelector("[data-comparison]")); });
  document.addEventListener("htmx:afterRequest", (event) => { const sequence = xhrSequences.get(event.detail.xhr); if (sequence === undefined || sequence === requestSequence) document.getElementById("results").setAttribute("aria-busy", "false"); });
  function showError(message) { const error = document.getElementById("request-error"); error.textContent = message; error.hidden = false; document.getElementById("results").setAttribute("aria-busy", "false"); }
  document.addEventListener("htmx:responseError", (event) => {
    const sequence = xhrSequences.get(event.detail.xhr); if (sequence !== undefined && sequence !== requestSequence) return;
    let message = "Comparison failed. Check the query and coordinates, then try again.";
    try { const response = JSON.parse(event.detail.xhr.responseText); if (typeof response.detail === "string") message = response.detail; else if (Array.isArray(response.detail)) message = response.detail.map((issue) => `${issue.loc.at(-1)}: ${issue.msg}`).join(" · "); } catch (error) { if (event.detail.xhr.status === 503) message = "This model is not available yet. Choose an available representation."; }
    showError(message);
  });
  document.addEventListener("htmx:sendError", (event) => { const sequence = xhrSequences.get(event.detail.xhr); if (sequence === undefined || sequence === requestSequence) showError("The server could not be reached."); });
})();
