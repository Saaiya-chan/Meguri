import { defineCollection, z } from 'astro:content';

const discussionsCollection = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    date: z.string(),
    description: z.string(),
    tags: z.array(z.string()).optional(),
  }),
});

export const collections = {
  discussions: discussionsCollection,
};
