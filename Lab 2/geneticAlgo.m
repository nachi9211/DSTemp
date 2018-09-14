%Data Science Matlab

function newpop = getnewpopulation(pop,score,nparentsratio,mutateprob)
% Generate a new population by first selecting the best performing
% chromosomes from the given pop matix, and subsequently generate new offspring chromosomes from randomly
% selected pairs of parent chromosomes.

% Step 1. Write code to select the top performing chromosomes. Use nparentsratio to
% calculate how many parents you need. If pop has 100 rows and
% nparentsration is 0.2, then you have to select the top performing 20
% chromosomes

[~,ind] = sort(score,'descend');
nparents = nparentsratio * size(pop,1);

newpop = zeros(size(pop));
newpop(1:nparents,:) = pop(ind(1:nparents),:);


% Step 2. Iterate until a new population is filled. Using the above
% example, you need to iterate 80 times. In each iteration create a new
% offspring chromosome from two randomly selected parent chromosomes. Use
% the function getOffSpring to generate a new offspring.

for j = nparents+1:size(pop,1)
    randparents = randperm(nparents);    
    newpop(j,:) = getOffSpring(newpop(randparents(1),:),newpop(randparents(2),:),mutateprob);    
end






function offspring = getOffSpring(parent1,parent2,mutateprob)
% Generate an offpsring from parent1 and parent2 and mutate the bits by
% using the probability mutateprob.

offspring = [];
% Step 1. Write code that generates one offspring from the given two parents

idx = ceil(rand(1)*(numel(parent1)-2))+1;
offspring = [parent1(1:idx),parent2(idx+1:end)];

% Step 2. Write code to mutate some bits with given mutation probability mutateprob

mutateidx = rand(1,numel(parent1)) < mutateprob;
offspring(mutateidx) = ~logical(offspring(mutateidx));

