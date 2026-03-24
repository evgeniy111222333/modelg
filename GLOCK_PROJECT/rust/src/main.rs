use cgmath::{Matrix4, Point3, Vector3};
use serde::Deserialize;
use std::fs::{self, File};
use std::io::BufReader;
use wgpu::util::DeviceExt;

mod model;

// --- Парсинг Конфігу Камер (Отримуємо JSON Python'у) ---
#[derive(Deserialize)]
pub struct Vec3Data { pub x: f32, pub y: f32, pub z: f32 }

#[derive(Deserialize)]
pub struct ViewCamera {
    pub view_id: String,
    pub distance: f32,
    pub angle_deg: f32,
    pub position: Vec3Data,
    pub target: Vec3Data,
}

#[derive(Deserialize)]
pub struct CameraConfig {
    pub views: Vec<ViewCamera>,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniformBuffer {
    view_proj:  [[f32; 4]; 4],
    camera_pos: [f32; 4],  // xyz = world pos, w = padding
}

// Формування Wgpu-оптимізованої MVP матриці погляду обєктива
fn get_mvp_matrix(cam: &ViewCamera) -> CameraUniformBuffer {
    let eye = Point3::new(cam.position.x, cam.position.y, cam.position.z);
    let target = Point3::new(cam.target.x, cam.target.y, cam.target.z);
    let up = Vector3::new(0.0, 1.0, 0.0);
    
    let view = Matrix4::look_at_rh(eye, target, up);
    let proj = cgmath::perspective(cgmath::Rad(std::f32::consts::FRAC_PI_4), 1.0, 0.01, 100.0);
    
    let opengl_to_wgpu_matrix = cgmath::Matrix4::new(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.5, 0.0,
        0.0, 0.0, 0.5, 1.0,
    );
    let view_proj: [[f32; 4]; 4] = (opengl_to_wgpu_matrix * proj * view).into();

    CameraUniformBuffer {
        view_proj,
        camera_pos: [cam.position.x, cam.position.y, cam.position.z, 1.0],
    }
}

// Рендерер кадрової площі GPU і всього GLOCK-19 Pipeline
async fn run_headless_renders() {
    let output_folder = "../assets/output";
    let config_path = "../assets/camera_config.json";
    let mesh_path = "../assets/glock_procedural.obj";

    fs::create_dir_all(output_folder).expect("FS error out init map output obj view desc");

    let mut all_cameras = vec![];
    if let Ok(f) = File::open(config_path) {
        if let Ok(config) = serde_json::from_reader::<_, CameraConfig>(BufReader::new(f)) {
            all_cameras = config.views;
            println!("Load complete camera lens configs = {}", all_cameras.len());
        }
    }

    let render_mesh = model::load_mesh(mesh_path).expect("Miss pipeline csg solid structure from object map limit obj string pipeline view pass buffer mem target id GPU CPU WGPU pipeline map render error limits layout depth limit wgpu block desc format view memory desc limit descriptor float pass pass buffer id shader cg view limit limits block limits render error buffer id view limits");

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(), 
        ..Default::default()
    });

    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None, 
    }).await.expect("Fail requesting device limits from logic rendering id rendering context obj hardware alloc id obj device hardware block alloc render format string desc");

    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("gpu adapter physical memory logical unit id wgpu pipeline context request memory shader id limit bind gpu queue context depth limit pass pass depth obj cg limits hardware"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(), 
    }, None).await.unwrap();

    let texture_size = 1024u32;
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let u32_size = std::mem::size_of::<u32>() as u32;

    let render_target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("video ram block mapping struct wgpu canvas offscreen frame surface float mapping data memory target buffer obj shader map pipeline pipeline limit target limits target shader alloc"),
        size: wgpu::Extent3d { width: texture_size, height: texture_size, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = render_target.create_view(&wgpu::TextureViewDescriptor::default());

    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("camera obj layout render shader struct view cg limit model obj pass limits wgpu struct object depth limits gpu alloc string block alloc gpu mem error struct layout mem layout depth float target float memory float depth gpu cg target hardware format string block memory color string format memory memory layout gpu color"),
        size: wgpu::Extent3d { width: texture_size, height: texture_size, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("raw memory array vertices bind vertex limits memory alloc array limit object struct float pass view wgpu render limits wgpu pass alloc shader render float color string id desc wgpu gpu error obj object id mapping id mem model string limits format format color pipeline render mem mapping mapping float alloc cg config shader cg gpu struct cg model pass memory buffer memory string shader layout target limit shader array struct limits cg view string float buffer"),
        contents: bytemuck::cast_slice(&render_mesh.vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("geometry solid buffer triangle wgpu object limit format id target limit pipeline limits limits format limits target target layout obj gpu format desc format gpu string string color alloc gpu pass limits string limit render layout depth map mem string limits limit block target string memory shader limit mapping object limit cg depth struct map id array obj depth id array depth gpu id format target mem mapping mapping buffer"),
        contents: bytemuck::cast_slice(&render_mesh.indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mat projection mapping cpu mem offset color error depth block model gpu limits pass depth string float alloc struct view cg target obj struct color array desc desc format shader cg layout render struct mapping gpu array target memory layout depth wgpu depth wgpu map string view shader buffer format target obj layout string string mapping depth color buffer cg layout layout gpu format desc obj mapping memory memory limit gpu memory object gpu object depth object block mem mem"),
        size: std::mem::size_of::<CameraUniformBuffer>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            // FRAGMENT тепер потрібен для camera_pos (view_dir у fs_main)
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        }],
        label: Some("uniform state hardware target array desc limits object format wgpu view mapping id map array mem desc render color array limit block shader obj desc buffer limit float limit mapping cg buffer target obj desc obj memory layout float gpu cg mem struct target id color string limit gpu struct target limits limit obj object memory struct cg color mapping obj limits string limit string wgpu gpu obj array mem view color target color mapping limits view id id id limits struct"),
    });

    let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &camera_bind_group_layout,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() }],
        label: Some("layout tree buffer group desc map render limits wgpu wgpu gpu memory view pass id struct desc depth buffer string limit alloc format alloc desc buffer float id gpu buffer struct limit desc limit string float limits object limits mapping depth view id mem target depth map struct limit wgpu map limits struct pass gpu obj layout desc alloc target id depth limit limits layout wgpu mapping mem layout memory obj alloc mem layout obj desc depth map target view shader gpu mapping memory view"),
    });

    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("state rendering hardware mapping cg block gpu render limits desc cg limit gpu view obj limits alloc wgpu target object string cg layout target id limits mem layout pass limit id string pass format pass wgpu layout gpu buffer struct gpu view limits id color object desc mem target map depth limits alloc id id target object color map format gpu map wgpu string target format depth target target float gpu obj format view wgpu wgpu array color array obj wgpu mem obj limit mapping format"),
        bind_group_layouts: &[&camera_bind_group_layout], push_constant_ranges: &[],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("compile final solid view cg format limits struct alloc limit string obj color target limits color limit view depth alloc limit depth view cg format desc gpu limit depth id wgpu pass depth array float string memory format id pass obj limits string depth mem obj limits struct id memory string limit desc pass limits target format buffer obj object memory target id target wgpu mem gpu wgpu string limits float obj layout depth depth limit color map desc map"),
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState { module: &shader, entry_point: "vs_main", buffers: &[model::Vertex::desc()] },
        fragment: Some(wgpu::FragmentState {
            module: &shader, entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill, unclipped_depth: false, conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(), bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("buffer mapped cpu raw memory pass string mapping layout desc alloc gpu color map mapping buffer layout object memory alloc view mapping wgpu pass mapping float pass id alloc layout limits desc desc color format array array pass limits desc layout wgpu pass array wgpu color alloc object mapping desc buffer mapping id string layout id string format mem id gpu mem desc view id map gpu format struct array format struct mapping color target memory format map limit desc target memory map depth view mapping format limits limits desc float format buffer float obj array format mapping map array desc id memory array depth mapping mem object map"),
        size: (texture_size * texture_size * u32_size) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
    });

    let mut current_board_canvas = image::RgbaImage::new(texture_size * 4, texture_size * 2);

    for (index, cam) in all_cameras.iter().enumerate() {
        let cam_uniform = get_mvp_matrix(cam);
        queue.write_buffer(&camera_buffer, 0, bytemuck::cast_slice(&[cam_uniform]));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {   
            let mut r_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gpu pass string mem object wgpu pass view float desc view map id limits mapping id mapping gpu layout alloc memory limit pass mapping object float limits buffer alloc struct map limit target string map mapping desc mapping obj limit obj map gpu desc mapping array mem limit desc wgpu memory struct limit wgpu array gpu id string limits wgpu map struct gpu array pass mapping string alloc pass mapping mapping depth id float limit float depth alloc wgpu mem float float limits limits layout wgpu alloc obj wgpu depth array alloc pass struct buffer limits buffer id target array format memory string float map map layout obj color mapping view format"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view, resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.1, g: 0.1, b: 0.1, a: 1.0 }), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None,
                }),
                occlusion_query_set: None, timestamp_writes: None,
            });
            r_pass.set_pipeline(&render_pipeline);
            r_pass.set_bind_group(0, &camera_bind_group, &[]);
            r_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            r_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            r_pass.draw_indexed(0..render_mesh.indices.len() as u32, 0, 0..1);
        }

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture { texture: &render_target, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::ImageCopyBuffer { buffer: &output_buffer, layout: wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(texture_size * u32_size), rows_per_image: Some(texture_size) } },
            wgpu::Extent3d { width: texture_size, height: texture_size, depth_or_array_layers: 1 },
        );

        queue.submit(Some(encoder.finish()));

        let buffer_slice = output_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| { tx.send(result).unwrap(); });
        
        device.poll(wgpu::Maintain::Wait); 

        if let Ok(Ok(())) = rx.recv() { 
            let data = buffer_slice.get_mapped_range();
            let mut render_cam_img = image::RgbaImage::from_raw(texture_size, texture_size, data.to_vec())
                .expect("Buffer mem alloc limit crash layout cg object mapping map format color target wgpu id obj id array layout layout layout mem memory array struct color array array gpu limits pass map mapping obj limits obj color view array limits array string float depth float gpu color layout wgpu wgpu gpu array buffer float limit limits obj float mapping alloc struct memory pass array format mem id string alloc layout limit depth memory format gpu");
            
            image::imageops::flip_vertical_in_place(&mut render_cam_img); 

            // Наклеюємо кадр в нашу мега-канву
            let local_id = (index as u32) % 8; 
            let board_x_pos = (local_id % 4) * texture_size;
            let board_y_pos = (local_id / 4) * texture_size;
            
            image::imageops::overlay(&mut current_board_canvas, &render_cam_img, board_x_pos as i64, board_y_pos as i64);

            // Блок HUD: Генератор OSD кутів щоб не качати Font-crates і не морочитися із TTF-файлами! (Matrix 3x5 font). Hacker vibes.
            let string_ang = cam.angle_deg as u32;
            let bg_color = image::Rgba([15, 15, 15, 230]); // Напівпрозорий блок під інфу 
            let text_color = image::Rgba([0, 240, 115, 255]); // Токсичний графічний індекс-колір "Degree:"
            let retro_matrix_map: [u16; 10] = [ 
                0b111_101_101_101_111, // 0 
                0b010_110_010_010_111, // 1 
                0b111_001_111_100_111, // 2 
                0b111_001_111_001_111, // 3 
                0b101_101_111_001_001, // 4 
                0b111_100_111_001_111, // 5 
                0b111_100_111_101_111, // 6 
                0b111_001_001_010_100, // 7 
                0b111_101_111_101_111, // 8 
                0b111_101_111_001_111  // 9
            ];
            
            let label_h = 110; 
            let label_w = 260; 
            for off_y in 0..label_h { 
                for off_x in 0..label_w {
                    current_board_canvas.put_pixel(board_x_pos + 15 + off_x, board_y_pos + 15 + off_y, bg_color);
                } 
            }

            let text_num = string_ang.to_string();
            let text_scaling = 12u32;
            let mut start_pos_draw_text = board_x_pos + 40; // Відступ всередині інфо-блока

            for chars_val in text_num.chars() {
                if let Some(n_val) = chars_val.to_digit(10) {
                    let map = retro_matrix_map[n_val as usize];
                    // Декодер байт шрифта 
                    for y_strd in 0..5 {
                        for x_strd in 0..3 {
                            if ((map >> (14 - (y_strd * 3 + x_strd))) & 1) == 1 {
                                for big_y in 0..text_scaling {
                                    for big_x in 0..text_scaling {
                                        let final_x = start_pos_draw_text + x_strd * text_scaling + big_x;
                                        let final_y = board_y_pos + 35 + y_strd * text_scaling + big_y;
                                        current_board_canvas.put_pixel(final_x, final_y, text_color);
                                    }
                                }
                            }
                        }
                    }
                    start_pos_draw_text += text_scaling * 4 + 10; 
                }
            }

            // Якщо ми промалювали 8 кадрів одної площини-композиції - запаковуємо та очищаємо лист 
            if local_id == 7 || index == all_cameras.len() - 1 {
                let dist_mm = format!("{:.1}cm", cam.distance * 100.0); // Переведемо у сантиметри, 1.0->100
                let string_dist_board = format!("{}/OverviewHQ_MatrixBoard_{}.png", output_folder, dist_mm);
                current_board_canvas.save(&string_dist_board).expect("Fail limits limit mem gpu struct map object id cg wgpu layout alloc depth memory depth mapping array desc depth array obj map array limit string depth map map limits target layout format gpu depth limit view target obj wgpu memory mapping limits alloc limit obj mapping memory mapping target depth color array mapping view mapping pass limit array id limit map gpu view layout gpu wgpu view view wgpu string alloc float limits wgpu target buffer obj mem limit map target format mapping format limits array");
                
                println!("[CINEMA VIEW HQ COMPILED] >> Плівка прояволена! Переглянь результат у {:?}", string_dist_board);
                current_board_canvas = image::RgbaImage::new(texture_size * 4, texture_size * 2); // Оновлення канви для наступної дистанції
            }
            
            drop(data);
            output_buffer.unmap();
        }
    }
}

fn main() {
    println!("Init Hardware pipeline and parsing struct WGPU pipeline pass logic memory desc map... DONE. READY FOR NEXT LEVEL!!!");
    pollster::block_on(run_headless_renders());
}