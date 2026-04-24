function memory = add_to_memory(new_sample_pts,new_subgrads,memory)

    memory.sample_pts = [memory.sample_pts,new_sample_pts];
    memory.subgrads = [memory.subgrads,new_subgrads];
    
    if(size(memory.sample_pts,2) > memory.max_size)
        memory.sample_pts = memory.sample_pts(:,end-memory.max_size+1:end);
        memory.subgrads = memory.subgrads(:,end-memory.max_size+1:end);
    end
end

