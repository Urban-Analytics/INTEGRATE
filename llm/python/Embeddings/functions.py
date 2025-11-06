def generate_sample_points(min_points_per_edge, max_points_per_edge, boundary_gdf, edges_gdf):
    # --------------------------------------------------------------------
    # Generate a sample of points along the street network (probabilistic)
    # --------------------------------------------------------------------

    # Re-project roads to a metric CRS for length calculations
    utm_crs = boundary_gdf.estimate_utm_crs()
    edges_proj = edges_gdf.to_crs(utm_crs)

    samples = []

    # Loop over every edge geometry
    for geom in edges_proj.geometry:
        if geom is None or geom.length == 0:
            continue

        length_m = geom.length
        expected = length_m / 1000 * density_per_km  # λ for Poisson
        n_points = np.random.poisson(expected)  # 0, 1, 2, …

        # Clip to limits
        n_points = max(min_points_per_edge, min(n_points, max_points_per_edge))

        if n_points == 0:
            continue

        # Evenly distribute interior points (skip endpoints)
        distances = np.linspace(0, length_m, n_points + 2)[1:-1]
        for d in distances:
            samples.append(geom.interpolate(d))

    # Build GeoDataFrame of sample points (back to WGS-84 for API use)
    points_gdf = (gpd.GeoDataFrame(geometry=gpd.GeoSeries(samples), crs=utm_crs)
        .to_crs(epsg=4326))
    points_gdf["lon"] = points_gdf.geometry.x.round(6)
    points_gdf["lat"] = points_gdf.geometry.y.round(6)
    points_gdf = points_gdf.drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)

    print( f"Generated {len(points_gdf)} points "
        f"with expected density {density_per_km} pts/km.")
