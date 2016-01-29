#![feature(type_ascription, vec_push_all, convert)]
#![allow(deprecated)]

extern crate image;
extern crate rand;
extern crate opencl;
extern crate num_cpus;
extern crate memmap;
extern crate time;
extern crate gtk;

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
use std::cell::RefCell;
use std::rc::Rc;
use memmap::{Mmap, Protection, MmapViewSync};
use time::PreciseTime;
use gtk::widgets::Builder;
use gtk::traits::*;
use gtk::{Window, Button, Box, FileChooserDialog, FileChooserAction, ResponseType, Entry, SpinButton};
use gtk::signal::Inhibit;

fn main() {
	if gtk::init().is_err() {
    	println!("Failed to initialize GTK.");
        return;
    }
	let glade_src = include_str!("GUI.glade");
    let builder = Builder::new_from_string(glade_src).unwrap();
	
	// populate OpenCL device list
	// get list of devices
	let platforms = opencl::hl::get_platforms();

	let mut devices = Vec::new();
	for platform in &platforms {
    	devices.push_all( platform.get_devices().as_slice() );
    }
	for device in &devices {
		println!( "Device: {}", device.name() );		
	}
	
	unsafe {
		let window: Window = builder.get_object("application_window").unwrap();
		let opencl_box: Box = builder.get_object("opencl_box").unwrap();
		let start_button: Button = builder.get_object("start_button").unwrap();
		let start_button = RefCell::new(start_button);		
		let output_directory_label: Entry = builder.get_object("directory_text").unwrap();
		let output_directory_label = RefCell::new(output_directory_label);
		let choose_output_directory_button: Button = builder.get_object("choose_output_directory_button").unwrap();
		
		window.connect_delete_event(|_, _| {
        	gtk::main_quit();
        	Inhibit(false)
        });
		
		// OpenCL list
		let mut check_buttons = Vec::new();
		for device in &devices {
			println!( "Device: {}", device.name() );	
			let mut name = device.name();
			name.pop();
			let check_button = gtk::CheckButton::new_with_label(&name).unwrap();
			opencl_box.pack_start(&check_button, true, true, 0);
			check_buttons.push( check_button );
		}
		let check_buttons = RefCell::new(check_buttons);
		
		// Start button update call
		let local_check_buttons = check_buttons.clone();
		let local_directory_label = output_directory_label.clone();
		let local_start_button = start_button.clone();
		let start_button_update = move || {
			// accessing reference to check_buttons
			let local_buttons = local_check_buttons.borrow_mut();
			let directory_label = local_directory_label.borrow();

			let mut start_possible = true;
			let label_text = directory_label.get_text().unwrap();
					
			if label_text.is_empty() {
				start_possible = false;
			}
			// traverse all checkbuttons
			let mut cl_start_possible = false;
			for local_button in local_buttons.iter() {
				// if one is set => true
				if local_button.get_active() {
					cl_start_possible = true;
					break;
				}
			}
			if !cl_start_possible {
				start_possible = false;
			}
					
			// if true => set start_button active else inactive
			let start_button = local_start_button.borrow_mut();
			start_button.set_sensitive( start_possible );
		};
		let start_button_update = Rc::new(start_button_update);
		
		// traverse check_buttons
		let buttons = check_buttons.borrow();
		for button in buttons.iter() {
			let own_closure = start_button_update.clone();
			button.connect_clicked( move |_| {
					own_closure();
				}
			);
		}
		
		// output directory handler
		let own_closure = start_button_update.clone();
		let directory_label = output_directory_label.clone();
		choose_output_directory_button.connect_clicked( move |_| {
				// create new file choose dialog
				let directory_dialog = 
					FileChooserDialog::new( "Choose Output Directory", 
											None, 
											FileChooserAction::CreateFolder, 
											dialog::buttons::OK_CANCEL );
				let result = directory_dialog.run();
				directory_dialog.hide();
				if result == ResponseType::Ok as i32 {
					// fetch directory path
					let directory = directory_dialog.get_filename().unwrap();
					// update label
					directory_label.borrow_mut().set_text(&directory);
					// call update funtion
					own_closure();
				}
			}
		);
		
		// Start button calls function
		let no_spheres_button: SpinButton = builder.get_object("no_spheres_button").unwrap();
		let xresbutton: SpinButton = builder.get_object("xresbutton").unwrap();
		let yresbutton: SpinButton = builder.get_object("yresbutton").unwrap();
		let zresbutton: SpinButton = builder.get_object("zresbutton").unwrap();
		let max_radius_button: SpinButton = builder.get_object("max_radius_button").unwrap();
		let background_color_button: SpinButton = builder.get_object("background_color_button").unwrap();
		let directory_label = output_directory_label.clone();
		start_button.borrow_mut().connect_clicked(move |_| {
				// fetch all parameters
				let num_spheres = no_spheres_button.get_value_as_int() as usize;
				let xres = xresbutton.get_value_as_int() as usize;
				let yres = yresbutton.get_value_as_int() as usize;
				let zres = zresbutton.get_value_as_int() as usize;
				let max_radius = max_radius_button.get_value_as_int();
				let background_color = background_color_button.get_value_as_int() as u8;
				let directory = directory_label.borrow().get_text().unwrap(); 
				do_calculation( num_spheres, xres, yres, zres, max_radius, background_color, directory );
        	}
		);
		
		window.show_all();
	}
	
	gtk::main();
}

fn do_calculation( num_spheres: usize, 
				   xres: usize, 
				   yres: usize,
				   zres: usize,
				   max_radius: i32,
				   background_color: u8,
				   directory: String ) {
	let mut temp_file = directory.clone();
	temp_file.push_str("/temp_file.bin");
    
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
	let mut xpos = Vec::new();
	let mut ypos = Vec::new();
	let mut zpos = Vec::new();
	let mut radius = Vec::new();
	let mut colors = Vec::new();
	{
		// open/create file
		let file_handle = File::create( &temp_file ).unwrap();
		// grow file
		file_handle.set_len( (xres*yres*zres) as u64 ).unwrap();
	} // close file by leaving scope
	// map read/write
	
	let mut volume_mmap = Mmap::open_path(&temp_file, Protection::ReadWrite).unwrap();
	{
		let mut volume: &mut [u8] = unsafe { volume_mmap.as_mut_slice() };
		for idx in 0..xres*yres*zres {
			volume[idx] = background_color;
		}
	}
	let mut volume_mmap_view = volume_mmap.into_view_sync();
	
	println!("Fill Buffers");
	for _ in 0..num_spheres {
		xpos.push(xbetween.ind_sample(&mut rng));
		ypos.push(ybetween.ind_sample(&mut rng));
		zpos.push(zbetween.ind_sample(&mut rng));
		radius.push(rbetween.ind_sample(&mut rng));
		colors.push(colorbetween.ind_sample(&mut rng));
		//colors.push(sphere_color);
	}
	
	// strip mutability
	let xpos = xpos;
	let ypos = ypos;
	let zpos = zpos;
	let radius = radius;
	let colors = colors;
    
    // OpenCL source
		// TODO: rewrite kernel and parameter passing to use 64bit addressing consistently
		let source = r#"
		__kernel void draw_spheres(
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
	
	// get list of devices
	let platforms = opencl::hl::get_platforms();

	let mut devices = Vec::new();
	for platform in &platforms {
    	devices.push_all( platform.get_devices().as_slice() );
    }
	for device in &devices {
		println!( "Device: {}", device.name() );
	}
	
	{
		let mut worker_pool = HashMap::new();
		let (request_tx, request_rx) = channel();
		
		// for loop to create threads
		for thread_idx in 1..devices.len(){
			let (command_tx, command_rx) = channel();
    		let local_request_tx = request_tx.clone();
    		let device = devices[thread_idx].clone();
    		let xpos = xpos.clone();
    		let ypos = ypos.clone();
    		let zpos = zpos.clone();
    		let radius = radius.clone();
    		let colors = colors.clone();
    		worker_pool.insert(thread_idx, (command_tx, thread::spawn( move || {
    							// open device
    							let context = device.create_context();
        						let queue = context.create_command_queue(&device);
        						// allocate buffers
        						let xbuf: CLBuffer<i32> = context.create_buffer(xpos.len(), opencl::cl::CL_MEM_READ_ONLY);
								let ybuf: CLBuffer<i32> = context.create_buffer(ypos.len(), opencl::cl::CL_MEM_READ_ONLY);
								let zbuf: CLBuffer<i32> = context.create_buffer(zpos.len(), opencl::cl::CL_MEM_READ_ONLY);
								let radiusbuf: CLBuffer<i32> = context.create_buffer(radius.len(), opencl::cl::CL_MEM_READ_ONLY);
								let colorbuf: CLBuffer<u8> = context.create_buffer(colors.len(), opencl::cl::CL_MEM_READ_ONLY);
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
								let dynbuf: CLBuffer<u8> = context.create_buffer(xres*yres*number_of_slices, opencl::cl::CL_MEM_READ_WRITE);
    							
    							
    							println!("Compile and set Kernel");
	
								let program = context.create_program_from_source(source);
								match program.build(&device) {
									Ok(v) => println!( "OpenCL program compiled ok: {}", v ),
									Err(e) => panic!( "OpenCL program could not be compiled: {}", e ),
								}
								let kernel = program.create_kernel("draw_spheres");

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
    							
    							// loop
    							loop {
    								//   request x slices
									local_request_tx.send((thread_idx, number_of_slices)).unwrap();
		    						// wait for answer
		    						let command: (usize, usize, Option<MmapViewSync>) = command_rx.recv().unwrap();
    								// if command == 0 => quit
    								if command.0 == command.1 {
	    								println!("Thread {} exiting.", thread_idx); 
    									break; 
    								}
    								// else work on package
    								let mut volume_mmap_view = command.2.unwrap();
    								//	calc loop parameters: zstart, zstop
    								let zbegin = command.0;
    								let zend = cmp::min(command.1, zres);

    								// write dynbuffer
    								{
    									let volume: &[u8] = unsafe { volume_mmap_view.as_slice() };
										queue.write(&dynbuf, &&volume[..], ());
									}				
    					
									// set kernel parameters
									kernel.set_arg(2, &zbegin);
									kernel.set_arg(3, &zend);
	
									println!("Queue kernel");
									let event = queue.enqueue_async_kernel(&kernel, num_spheres, None, ());
	
									println!("Read back volume");
									{
										let mut volume: &mut [u8] = unsafe { volume_mmap_view.as_mut_slice() };
										queue.read(&dynbuf, &mut &mut volume[..], &event);
									}
    							}
    						})));
		}
		
		// while loop to distribute work
		let starttime = PreciseTime::now(); 
		let mut zlevel = 0;
		while zlevel < zres {
			// listen on the request channel
    		let (thread_idx, no_slices) = request_rx.recv().unwrap();
    		let offset = mem::size_of::<u8>().checked_mul(cmp::min(no_slices, zres-zlevel)*xres*yres as usize).unwrap();
    		let (send_mmap, volume_mmap_view2) = volume_mmap_view.split_at(offset).unwrap();
    		volume_mmap_view = volume_mmap_view2;
    		// send next z value to the first thread requesting work
    		let worker = worker_pool.get(&thread_idx).unwrap();
    		worker.0.send((zlevel, zlevel + no_slices, Some(send_mmap))).unwrap();
    		zlevel += no_slices;
		}
		
		println!("Waiting for threads.");
		
		// for i over worker_pool.size()
    	while !worker_pool.is_empty() {
    		// listen on the request channel
    		let (thread_idx, _) = request_rx.recv().unwrap();
    		println!("Thread {} requests new work, needs to quit.", thread_idx);
    		// send terminate signal + join on the thread handle
    		let worker_handle = worker_pool.remove(&thread_idx).unwrap();
    		worker_handle.0.send((0, 0, None)).unwrap();
    		let _ = worker_handle.1.join();
    	}
    	let endtime = PreciseTime::now();
    	println!("Loop took {} seconds.", starttime.to(endtime));
	}
	
		
	
	println!("Write PNGs");
	let no_cpus = num_cpus::get();
	// spawn threads
	// each thread sends a ready signal to main thread
	// main thread responds either with a - all_done_finish_up or with a
	//                                    - here is the new work package
	//                                    (quit: bool, z: usize)
	// threads work through the packages until receiving the finish_up signal
	
	
	let volume_mmap = Mmap::open_path(&temp_file, Protection::ReadWrite).unwrap();
	
	let volume_mmap_handle = Arc::new( volume_mmap );
    let (request_tx, request_rx) = channel();
    
    let mut worker_pool = HashMap::new();
    for thread_idx in 0..no_cpus {
    	let (command_tx, command_rx) = channel();
    	let local_request_tx = request_tx.clone();
    	let child_mmap = volume_mmap_handle.clone();
    	let directory_clone = directory.clone();
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
    						let path = directory_clone.clone() + &"/slice".to_string() + &command.1.to_string() + &".png".to_string();
    						let ref mut fout = File::create(&Path::new( &*path)).unwrap();
    	
				    		let _ = image::ImageLuma8(newimage).save(fout, image::PNG);
    					}
    					})));
    }
    
    // for loop over all z values
    for z in 0..zres {
    	// listen on the request channel
    	let thread_idx = request_rx.recv().unwrap();
    	// send next z value to the first thread requesting work
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
    }
}
