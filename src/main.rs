#![feature(type_ascription, step_by)]

// TODO: mem-mapped volume
// TODO: use multiple OpenCL devices in parallel - careful about memory usage

extern crate image;
extern crate rand;
extern crate opencl;
extern crate num_cpus;
extern crate memmap;

use opencl::mem::CLBuffer;

use rand::{SeedableRng, StdRng};
use rand::distributions::{IndependentSample, Range};

use std::thread;
use std::sync::mpsc::channel;
use std::sync::Arc;
use std::fs::File;
use std::path::Path;
use std::mem;
use std::cmp;
use std::collections::HashMap;
use memmap::{Mmap, Protection};


fn main() {
    println!("Set parameters");
    // set parameters
    let num_balls: usize = 10000;
    let max_radius = 40;
    let xres: usize = 5000;
    let yres: usize = 5000;
    let zres: usize = 100;
    
    let background_color: u8 = 255;
//    let ball_color: u8 = 0;
    
    let seed: &[_] = &[1, 2, 3, 4];
    let mut rng: StdRng = SeedableRng::from_seed(seed);
    let xbetween = Range::new(0:i32, xres as i32);
    let ybetween = Range::new(0:i32, yres as i32);
    let zbetween = Range::new(0:i32, zres as i32);
    let rbetween = Range::new(0:i32, max_radius);
    let colorbetween = Range::new(0:u8, 255:u8);
    
    // prep buffers
    println!("Prep Buffers");
    // Set up our work dimensions / data set size:
//    let xpos = SimpleDims::OneDim(num_balls);
	let mut xpos = Vec::new();
	let mut ypos = Vec::new();
	let mut zpos = Vec::new();
	let mut radius = Vec::new();
	let mut colors = Vec::new();
	{
		// open/create file
		let file_handle = File::create("temp_file.bin").unwrap();
		// grow file
		file_handle.set_len( (xres*yres*zres) as u64 ).unwrap();
	} // close file by leaving scope
	// map read/write
	//let statbuf = Arc::new(volume);
	
	let mut volume_mmap = Mmap::open_path("temp_file.bin", Protection::ReadWrite).unwrap();
	// associate with slice handle
	//let mut volume: &[u8] = unsafe { stat_volume_mmap.as_slice() };
	//let mut volume = vec![background_color; xres*yres*zres];
	{
		let mut volume: &mut [u8] = unsafe { volume_mmap.as_mut_slice() };
		for idx in 0..xres*yres*zres {
			volume[idx] = background_color;
		}
	}
	
	println!("Fill Buffers");
	for _ in 0..num_balls {
		xpos.push(xbetween.ind_sample(&mut rng));
		ypos.push(ybetween.ind_sample(&mut rng));
		zpos.push(zbetween.ind_sample(&mut rng));
		radius.push(rbetween.ind_sample(&mut rng));
		colors.push(colorbetween.ind_sample(&mut rng));
		//colors.push(ball_color);
	}
    
    // OpenCL source
//	let source = r#"
//		__kernel void draw_balls(
//					  __private unsigned long const xres,
//					  __private unsigned long const yres,
//					  __private unsigned long const zres,
//					  __global int const* const xpos,
//					  __global int const* const ypos,
//					  __global int const* const zpos,
//					  __global int const* const radius,
//					  __global uchar const* const color,
//					  __global uchar* const buffer )
//		{
//			size_t idx = get_global_id(0);
//			
//			int sqradius = radius[idx]*radius[idx];			
//			// tripple loop
//			for( int z = max(0, zpos[idx]-radius[idx]-1); z < min((int)zres, zpos[idx]+radius[idx]+1); z++ ){
//				for( int y = max(0, ypos[idx]-radius[idx]-1); y < min((int)yres, ypos[idx]+radius[idx]+1); y++ ){
//					for( int x = max(0, xpos[idx]-radius[idx]-1); x < min((int)xres, xpos[idx]+radius[idx]+1); x++ ){
//						int dx = xpos[idx]-x;
//						int dy = ypos[idx]-y;
//						int dz = zpos[idx]-z;
//						int sqdistance = dx*dx + dy*dy + dz*dz;
//						if( sqdistance < sqradius ){
//							buffer[x + y * xres + z * xres * yres] = color[idx]; 
//						}
//					}
//				}
//			}
//		}
//		"#;
		
		// TODO: color priorities
		// TODO: rewrite kernel and parameter passing to use 64bit addressing consistently
		let source = r#"
		__kernel void draw_balls(
					  __private unsigned long const xres,
					  __private unsigned long const yres,
					  __private unsigned long const zbegin,
					  __private unsigned long const zend,
					  __global int const* const xpos,
					  __global int const* const ypos,
					  __global int const* const zpos,
					  __global int const* const radius,
					  __global uchar const* const color,
					  __global uchar* const buffer )
		{
			size_t idx = get_global_id(0);
			
			int sqradius = radius[idx]*radius[idx];			
			// tripple loop
			for( int z = max((int)zbegin, zpos[idx]-radius[idx]-1); z < min((int)zend, zpos[idx]+radius[idx]+1); z++ ){
				for( int y = max(0, ypos[idx]-radius[idx]-1); y < min((int)yres, ypos[idx]+radius[idx]+1); y++ ){
					for( int x = max(0, xpos[idx]-radius[idx]-1); x < min((int)xres, xpos[idx]+radius[idx]+1); x++ ){
						int dx = xpos[idx]-x;
						int dy = ypos[idx]-y;
						int dz = zpos[idx]-z;
						int sqdistance = dx*dx + dy*dy + dz*dz;
						if( sqdistance < sqradius && 
							buffer[x + y * xres + (z-zbegin) * xres * yres] > color[idx] ){
							buffer[x + y * xres + (z-zbegin) * xres * yres] = color[idx]; 
						}
					}
				}
			}
		}
	"#;
	
	let (device, ctx, queue) = opencl::util::create_compute_context().unwrap();	

	println!("Prep OpenCL buffers");
	let xbuf: CLBuffer<i32> = ctx.create_buffer(xpos.len(), opencl::cl::CL_MEM_READ_ONLY);
	let ybuf: CLBuffer<i32> = ctx.create_buffer(ypos.len(), opencl::cl::CL_MEM_READ_ONLY);
	let zbuf: CLBuffer<i32> = ctx.create_buffer(zpos.len(), opencl::cl::CL_MEM_READ_ONLY);
	let radiusbuf: CLBuffer<i32> = ctx.create_buffer(radius.len(), opencl::cl::CL_MEM_READ_ONLY);
	let colorbuf: CLBuffer<u8> = ctx.create_buffer(colors.len(), opencl::cl::CL_MEM_READ_ONLY);
	
	// check max mem
	let total_mem = device.global_mem_size();
	let max_block_size = device.max_mem_alloc_size();
	// subtract the other buffers
	let stat_buffer_size =  xpos.len() * mem::size_of::<i32>() +
							ypos.len() * mem::size_of::<i32>() + 
							zpos.len() * mem::size_of::<i32>() +
							radius.len() * mem::size_of::<i32>() + 
							colors.len() *  mem::size_of::<u8>();
	// size the buffer right
	let dyn_buffer_size = cmp::min(total_mem as usize- stat_buffer_size, max_block_size as usize);
	// allocate
	let number_of_slices = cmp::min(dyn_buffer_size / (xres*yres*mem::size_of::<u8>()), zres);
	assert!(number_of_slices > 0);
	//let mut dyn_volume = vec![background_color; xres*yres*number_of_slices];
	let dynbuf: CLBuffer<u8> = ctx.create_buffer(xres*yres*number_of_slices, opencl::cl::CL_MEM_READ_WRITE);

	println!("Compile and set Kernel");
	
	let program = ctx.create_program_from_source(source);
	match program.build(&device) {
		Ok(v) => println!( "OpenCL program compiled ok: {}", v ),
		Err(e) => panic!( "OpenCL program could not be compiled: {}", e ),
	}
	let kernel = program.create_kernel("draw_balls");

	println!("Set static kernel parameters");
	kernel.set_arg(0, &xres);
	kernel.set_arg(1, &yres);
	//kernel.set_arg(2, &zbegin);
	//kernel.set_arg(3, &zend);
	kernel.set_arg(4, &xbuf);
	kernel.set_arg(5, &ybuf);
	kernel.set_arg(6, &zbuf);
	kernel.set_arg(7, &radiusbuf);
	kernel.set_arg(8, &colorbuf);
	kernel.set_arg(9, &dynbuf);

	println!("Push static buffers");
	
	queue.write(&xbuf, &&xpos[..], ());
	queue.write(&ybuf, &&ypos[..], ());
	queue.write(&zbuf, &&zpos[..], ());
	queue.write(&radiusbuf, &&radius[..], ());
	queue.write(&colorbuf, &&colors[..], ());
	//queue.write(&outbuf, &&volume[..], ());

	{
		let mut volume: &mut [u8] = unsafe { volume_mmap.as_mut_slice() };
		
		// loop starts here
		for slice_begin in (0..zres).step_by(number_of_slices) {
			//	calc loop parameters: zstart, zstop
			let zbegin = slice_begin;
			let zend = cmp::min(slice_begin + number_of_slices, zres);
			println!("Calc zslices from {} to {}", zbegin, zend);
			// clean dynbuffer
	//		for idx in 0..dyn_volume.len() {
	//			dyn_volume[idx] = background_color;
	//		}
		
			// write dynbuffer
			queue.write(&dynbuf, &&volume[zbegin*xres*yres..zend*xres*yres], ());
		
			// set kernel parameters
			kernel.set_arg(2, &zbegin);
			kernel.set_arg(3, &zend);
	
			println!("Queue kernel");
			let event = queue.enqueue_async_kernel(&kernel, num_balls, None, ());
	
			println!("Read back volume");
			queue.read(&dynbuf, &mut &mut volume[zbegin*xres*yres..zend*xres*yres], &event);
		
	//		println!("Copy data to big buffer");
	//		// copy data to big buffer
	//		for idx in 0..(zend-zbegin)*xres*yres {
	//			volume[zbegin*xres*yres+idx] = dyn_volume[idx];
	//		}
	//		//std::slice::bytes::copy_memory(&dyn_volume[..], &mut volume[zbegin*xres*yres..zend*xres*yres]);
	//		println!("Data Copy done");
		}
	}
	
	println!("Write PNGs");
	let no_cpus = num_cpus::get();
	// spawn threads
	// each thread sends a ready signal to main thread
	// main thread responds either with a - all_done_finish_up or with a
	//                                    - here is the new work package
	//                                    (quit: bool, z: usize)
	// threads work through the packages until receiving the finish_up signal
	
	//let statbuf = Arc::new(volume);
	let volume_mmap_handle = Arc::new(volume_mmap);
    let (request_tx, request_rx) = channel();
    
    //let mut worker_pool = Vec::new();// think about how to spawn threads iter().??;
    let mut worker_pool = HashMap::new();
    for thread_idx in 0..no_cpus {
    	let (command_tx, command_rx) = channel();
    	let local_request_tx = request_tx.clone();
    	let child_mmap = volume_mmap_handle.clone();
    	//let worker = (command_tx, thread::spawn( move || {
    	worker_pool.insert( thread_idx, (command_tx, thread::spawn( move || {
    					let data: &[u8] = unsafe { child_mmap.as_slice() };
    					loop {
    						// send true over the request channel to ask for new work package
    						local_request_tx.send(thread_idx).unwrap();
    						// wait for answer
    						let command: (bool, usize) = command_rx.recv().unwrap();
    						// if quit => quit
    						if !command.0 {
	    						println!("Thread {} exiting.", thread_idx); 
    							break; 
    						}
    						// else work on package
    						let mut newimage = image::ImageBuffer::new( xres as u32, yres as u32 );
							for (x, y, pixel) in newimage.enumerate_pixels_mut() {
	    						*pixel = image::Luma( [data[x as usize + (y as usize)*xres + (command.1 as usize)*xres*yres] as u8] );
    						}
    						let path = "slice".to_string() + &command.1.to_string() + &".png".to_string();
    						let ref mut fout = File::create(&Path::new( &*path)).unwrap();
    	
				    		let _ = image::ImageLuma8(newimage).save(fout, image::PNG);
    					}
    					})));
    	//worker_pool.push( worker );
    }
    
    // for loop over all z values
    for z in 0..zres {
    	// listen on the request channel
    	let thread_idx = request_rx.recv().unwrap();
    	// send next z value to the first thread requesting work
    	//worker_pool[thread_idx].0.send((true, z)).unwrap();
    	let worker = worker_pool.get(&thread_idx).unwrap();
    	worker.0.send((true, z)).unwrap();
    }
    
    // for i over worker_pool.size()
    while !worker_pool.is_empty() {
    	// listen on the request channel
    	let thread_idx = request_rx.recv().unwrap();
    	// send terminate signal + join on the thread handle
    	let worker_handle = worker_pool.remove(&thread_idx).unwrap();
    	worker_handle.0.send((false,0)).unwrap();
    	let _ = worker_handle.1.join();
    	//let thread_ref = &worker_pool[thread_idx].1;
    	//let _ = thread_ref.join();
    }
    
//    let mut pool = Vec::new();
//    for z in 0..zres {
//    	let data = statbuf.clone();
//    	pool.push( thread::spawn( move || {
//    			let mut newimage = image::ImageBuffer::new( xres as u32, yres as u32 );
//				for (x, y, pixel) in newimage.enumerate_pixels_mut() {
//    				*pixel = image::Luma( [data[x as usize + (y as usize)*xres + (z as usize)*xres*yres] as u8] );
//    			}
//    			let path = "slice".to_string() + &z.to_string() + &".png".to_string();
//    			let ref mut fout = File::create(&Path::new( &*path)).unwrap();
//    	
//		    	let _ = image::ImageLuma8(newimage).save(fout, image::PNG);
//    	}) );
//    }
    
//    for _ in 0..zres {
//    	let _ = pool.pop().unwrap().join();
//    }
}
