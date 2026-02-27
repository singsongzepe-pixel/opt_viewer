
-- project info
set_project("opt_viewer")
set_version("1.0.0")

-- language
set_languages("c++17")
add_requires("tinygltf")

-- check the graphic cards
-- target("check")
--     set_kind("binary")
--     add_files("src/check.cpp")

--     -- add_rules("utils.symbols.export_all")
--     add_links("vulkan-1")

--     if os.getenv("VULKAN_SDK") then
--         add_includedirs(os.getenv("VULKAN_SDK") .. "/Include")
--         add_linkdirs(os.getenv("VULKAN_SDK") .. "/Lib")
--     end

-- demo 1
-- target("demo1")
--     set_kind("binary")
    
--     add_rules("utils.glsl2spv")
--     add_files("src/demo1.cpp")

--     before_build(function (target)
--         import("core.project.depend")

--         local bin_dir = target:targetdir()
--         local shader_out_dir = path.join(bin_dir, "shaders")
--         if not os.isdir(shader_out_dir) then os.mkdir(shader_out_dir) end

--         for _, shader_path in ipairs(os.files("shaders/*.comp")) do
--             local filename = path.filename(shader_path)
--             local output_path = path.join(shader_out_dir, filename .. ".spv")

--             depend.on_changed(function ()
--                 print("Compiling shader: %s -> %s", shader_path, output_path)
--                 os.runv("glslangValidator", {"-V", shader_path, "-o", output_path})
--             end, {files = shader_path})
--         end
--     end)

--     set_rundir("$(builddir)/$(plat)/$(arch)/$(mode)")

--     add_links("vulkan-1")
--     if os.getenv("VULKAN_SDK") then
--         add_includedirs(os.getenv("VULKAN_SDK") .. "/Include")
--         add_linkdirs(os.getenv("VULKAN_SDK") .. "/Lib")
--     end

-- demo 2 
-- target("demo2")
--     set_kind("binary")
    
--     add_includedirs("thirdparty/glfw-3.4/include")
--     add_linkdirs("thirdparty/glfw-3.4/lib-vc2022")
--     add_links("glfw3")

--     if is_plat("windows") then 
--         -- for link libraries of os itself
--         add_syslinks("user32", "gdi32", "shell32")
--     end

--     add_packages("glm", "vulkan-sdk")

--     add_rules("utils.glsl2spv")
--     add_files("src/demo2.cpp")

--     before_build(function (target)
--         import("core.project.depend")

--         local bin_dir = target:targetdir()
--         local shader_out_dir = path.join(bin_dir, "shaders")
--         if not os.isdir(shader_out_dir) then os.mkdir(shader_out_dir) end

--         local files = os.files("shaders/*.vert")
--         table.join2(files, os.files("shaders/*.frag"))

--         for _, shader_path in ipairs(files) do
--             local filename = path.filename(shader_path)
--             local output_path = path.join(shader_out_dir, filename .. ".spv")

--             depend.on_changed(function ()
--                 print("Compiling shader: %s -> %s", shader_path, output_path)
--                 os.runv("glslangValidator", {"-V", shader_path, "-o", output_path})
--             end, {files = shader_path})
--         end
--     end)

--     set_rundir(path.join("$(builddir)", "$(plat)", "$(arch)", "$(mode)"))

--     if is_plat("windows") then
--         add_links("vulkan-1")
--     end

--     if os.getenv("VULKAN_SDK") then
--         add_includedirs(os.getenv("VULKAN_SDK") .. "/Include")
--         add_linkdirs(os.getenv("VULKAN_SDK") .. "/Lib")
--     end

-- demo 3
-- Blinn-Phong model
target("demo3")
    set_kind("binary")
    
    add_includedirs("thirdparty/glfw-3.4/include")
    add_linkdirs("thirdparty/glfw-3.4/lib-vc2022")
    add_links("glfw3")

    if is_plat("windows") then 
        -- for link libraries of os itself
        add_syslinks("user32", "gdi32", "shell32")
    end

    add_packages("glm", "vulkan-sdk")

    add_rules("utils.glsl2spv")
    add_files("src/demo3.cpp")

    before_build(function (target)
        import("core.project.depend")

        local bin_dir = target:targetdir()
        local shader_out_dir = path.join(bin_dir, "shaders")
        if not os.isdir(shader_out_dir) then os.mkdir(shader_out_dir) end

        local files = os.files("shaders/*.vert")
        table.join2(files, os.files("shaders/*.frag"))

        for _, shader_path in ipairs(files) do
            local filename = path.filename(shader_path)
            local output_path = path.join(shader_out_dir, filename .. ".spv")

            depend.on_changed(function ()
                print("Compiling shader: %s -> %s", shader_path, output_path)
                os.runv("glslangValidator", {"-V", shader_path, "-o", output_path})
            end, {files = shader_path})
        end
    end)

    set_rundir(path.join("$(builddir)", "$(plat)", "$(arch)", "$(mode)"))

    if is_plat("windows") then
        add_links("vulkan-1")
    end

    if os.getenv("VULKAN_SDK") then
        add_includedirs(os.getenv("VULKAN_SDK") .. "/Include")
        add_linkdirs(os.getenv("VULKAN_SDK") .. "/Lib")
    end


-- demo 4
-- PBR model
target("demo4")
    set_kind("binary")
    
    add_includedirs("thirdparty/glfw-3.4/include")
    add_linkdirs("thirdparty/glfw-3.4/lib-vc2022")
    add_links("glfw3")

    if is_plat("windows") then 
        -- for link libraries of os itself
        add_syslinks("user32", "gdi32", "shell32")
    end

    add_packages("tinygltf")
    add_packages("glm", "vulkan-sdk")

    add_rules("utils.glsl2spv")
    add_files("src/demo4.cpp")

    before_build(function (target)
        import("core.project.depend")

        local bin_dir = target:targetdir()
        local shader_out_dir = path.join(bin_dir, "shaders")
        if not os.isdir(shader_out_dir) then os.mkdir(shader_out_dir) end

        local files = os.files("shaders/*.vert")
        table.join2(files, os.files("shaders/*.frag"))

        for _, shader_path in ipairs(files) do
            local filename = path.filename(shader_path)
            local output_path = path.join(shader_out_dir, filename .. ".spv")

            depend.on_changed(function ()
                print("Compiling shader: %s -> %s", shader_path, output_path)
                os.runv("glslangValidator", {"-V", shader_path, "-o", output_path})
            end, {files = shader_path})
        end
    end)

    set_rundir(path.join("$(builddir)", "$(plat)", "$(arch)", "$(mode)"))

    if is_plat("windows") then
        add_links("vulkan-1")
    end

    if os.getenv("VULKAN_SDK") then
        add_includedirs(os.getenv("VULKAN_SDK") .. "/Include")
        add_linkdirs(os.getenv("VULKAN_SDK") .. "/Lib")
    end

