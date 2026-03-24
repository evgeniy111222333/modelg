use tobj;
use bytemuck::{Pod, Zeroable};

// Упаковка структури вершин під залізо пам'яті (щоб GPU її не фрагментувала)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position:[f32; 3],
    pub normal: [f32; 3],
}

impl Vertex {
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3, // збігається з @location(0) в шейдері
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3, // збігається з @location(1) в шейдері
                },
            ],
        }
    }
}

pub struct RenderModel {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

pub fn load_mesh(obj_file_path: &str) -> Option<RenderModel> {
    println!("Ініційовано хардкорне перенесення вершин Python-SDF...");
    
    let load_options = tobj::LoadOptions {
        single_index: true,
        triangulate: true,
        ..Default::default()
    };
    
    match tobj::load_obj(obj_file_path, &load_options) {
        Ok((models, _materials)) => {
            if let Some(mesh_group) = models.first() {
                let m = &mesh_group.mesh;
                
                let mut output_vertices = Vec::new();
                let vertex_count = m.positions.len() / 3;
                let has_normals = !m.normals.is_empty();
                
                for i in 0..vertex_count {
                    let normal = if has_normals {
                        [
                            m.normals[i * 3],
                            m.normals[i * 3 + 1],
                            m.normals[i * 3 + 2],
                        ]
                    } else {
                        // Якщо нормаль загубилась в парсингу CSG - стріляємо просто "вгору"
                        [0.0, 1.0, 0.0]
                    };

                    output_vertices.push(Vertex {
                        position:[
                            m.positions[i * 3],
                            m.positions[i * 3 + 1],
                            m.positions[i * 3 + 2],
                        ],
                        normal,
                    });
                }

                println!("(OK) GPU: Модель на {} полігонів імпортована успішно.", m.indices.len() / 3);

                return Some(RenderModel {
                    vertices: output_vertices,
                    indices: m.indices.clone(),
                });
            }
        }
        Err(err) => eprintln!("ПОМИЛКА PARSE ОБ'ЄКТА ПІСТОЛЕТА: {:?}", err),
    }
    None
}